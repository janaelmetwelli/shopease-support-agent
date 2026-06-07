"""
DocumentIndexer — loads all knowledge sources, chunks them, embeds them,
and persists them in ChromaDB.

Run once (or after data changes) with:
    python scripts/index_documents.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from rag.embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)

# Collection names
COLLECTION_PRODUCTS        = "product_catalog"
COLLECTION_FAQS            = "faqs"
COLLECTION_POLICIES        = "policies"
COLLECTION_MANUALS         = "product_manuals"
COLLECTION_COSMETICS       = "cosmetics_catalog"
COLLECTION_RECOMMENDATIONS = "recommendations"
COLLECTION_STORE           = "store_info"
COLLECTION_ALL             = "all_docs"   # unified collection for hybrid search


class DocumentIndexer:

    def __init__(self):
        self.embedder = LocalEmbeddings(model_name=settings.embedding_model)
        self.client   = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.data_dir = Path("./data")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def _upsert(self, collection, ids, docs, metas):
        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=self.embedder.embed_documents(docs),
            metadatas=metas,
        )
        logger.info("Upserted %d docs into '%s'", len(ids), collection.name)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        ).split_text(text)

    def _read_json(self, filename: str):
        return json.loads((self.data_dir / filename).read_text(encoding="utf-8"))

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_products(self):
        ids, docs, metas = [], [], []
        for p in self._read_json("product_catalog.json"):
            ids.append(f"product_{p['product_id']}")
            docs.append(
                f"Product: {p['name']}\n"
                f"Category: {p['category']}\n"
                f"Price: ${p['price']:.2f}\n"
                f"Description: {p['description']}\n"
                f"Warranty: {p['warranty_years']} year(s)\n"
                f"In Stock: {'Yes' if p['in_stock'] else 'No'}\n"
                f"SKU: {p['sku']}"
            )
            metas.append({
                "source":     "product_catalog",
                "product_id": p["product_id"],
                "category":   p["category"],
                "price":      p["price"],
                "in_stock":   str(p["in_stock"]),
            })
        return ids, docs, metas

    def _load_faqs(self):
        ids, docs, metas = [], [], []
        for faq in self._read_json("faqs.json"):
            ids.append(f"faq_{faq['id']}")
            docs.append(f"Question: {faq['question']}\nAnswer: {faq['answer']}")
            metas.append({"source": "faq", "faq_id": faq["id"], "category": faq["category"]})
        return ids, docs, metas

    def _load_policies(self):
        ids, docs, metas = [], [], []
        for policy_name, filename in [
            ("shipping_policy", "shipping_policy.md"),
            ("returns_policy",  "returns_policy.md"),
        ]:
            chunks = self._chunk_text(
                (self.data_dir / filename).read_text(encoding="utf-8")
            )
            for i, chunk in enumerate(chunks):
                ids.append(f"{policy_name}_chunk_{i}")
                docs.append(chunk)
                metas.append({"source": policy_name, "chunk_index": i, "total_chunks": len(chunks)})
        return ids, docs, metas

    def _load_manuals(self):
        """Semantic chunking — one doc per individual step / issue / tip."""
        ids, docs, metas = [], [], []
        sections = [
            ("step_by_step_usage",   "usage",          "How to use"),
            ("troubleshooting_tips", "troubleshooting", "Troubleshooting"),
            ("maintenance_tips",     "maintenance",     "Maintenance"),
        ]
        for m in self._read_json("product_manuals.json"):
            pid      = m.get("product_id", "UNKNOWN")
            name     = m.get("product_name", "Unknown Product")
            brand    = m.get("brand", "")
            category = m.get("category", "")
            for field, section_key, section_label in sections:
                items = m.get(field, [])
                for i, item in enumerate(items):
                    ids.append(f"manual_{section_key}_{pid}_{i}")
                    docs.append(f"{name} ({brand}) — {section_label}:\n{item}")
                    metas.append({
                        "source":       "product_manuals",
                        "product_id":   pid,
                        "product_name": name,
                        "brand":        brand,
                        "category":     category,
                        "section":      section_key,
                        "item_index":   i,
                        "total_items":  len(items),
                    })
        return ids, docs, metas

    def _load_cosmetics(self):
        ids, docs, metas = [], [], []
        for item in self._read_json("cosmetics_catalog.json"):
            pid      = item.get("product_id", "COS-XXX")
            name     = item.get("product_name", "Unknown")
            brand    = item.get("brand", "")
            category = item.get("subcategory", item.get("category", ""))
            skin_type    = ", ".join(item.get("skin_type", []))
            ingredients  = ", ".join(item.get("ingredients", []))
            price        = item.get("price", 0)
            ids.append(f"cosmetic_{pid}")
            docs.append(
                f"Product: {name}\nBrand: {brand}\nCategory: {category}\n"
                f"Skin/Hair Type: {skin_type}\n"
                f"Key Benefits: {item.get('key_benefits', '')}\n"
                f"How to Use: {item.get('how_to_use', '')}\n"
                f"Ingredients: {ingredients}\n"
                f"Price: ${price:.2f}"
            )
            metas.append({
                "source":     "cosmetics_catalog",
                "product_id": pid,
                "category":   category,
                "brand":      brand,
                "price":      price,
                "skin_type":  skin_type,
            })
        return ids, docs, metas

    def _load_recommendations(self):
        ids, docs, metas = [], [], []
        data = self._read_json("recommendations.json")

        # Bundles — one doc per bundle
        for b in data.get("product_bundles", []):
            ids.append(f"bundle_{b['bundle_id']}")
            docs.append(
                f"Bundle: {b['name']}\n"
                f"Description: {b['description']}\n"
                f"Products included: {', '.join(b.get('products', []))}\n"
                f"Bundle price: ${b.get('bundle_price', 0):.2f} (save ${b.get('savings', 0):.2f})\n"
                f"Tags: {', '.join(b.get('tags', []))}"
            )
            metas.append({"source": "recommendations", "type": "bundle", "bundle_id": b["bundle_id"]})

        # Trending items — one doc per item
        for t in data.get("trending_items", []):
            ids.append(f"trending_{t['product_id']}")
            docs.append(
                f"Trending #{t['rank']} at ShopEase: {t['product_name']}\n"
                f"Category: {t['category']}\n"
                f"Why it's popular: {t['reason']}"
            )
            metas.append({
                "source":       "recommendations",
                "type":         "trending",
                "product_id":   t["product_id"],
                "product_name": t["product_name"],
                "rank":         t["rank"],
                "category":     t["category"],
            })

        # Seasonal offers — one doc per offer
        for offer in data.get("seasonal_offers", []):
            ids.append(f"offer_{offer['offer_id']}")
            docs.append(
                f"Seasonal Offer: {offer['name']}\n"
                f"Description: {offer['description']}\n"
                f"Discount: {offer.get('discount_percent', '')}% off\n"
                f"Promo code: {offer.get('promo_code', 'N/A')}\n"
                f"Applicable products: {', '.join(offer.get('applicable_products', []))}\n"
                f"Valid until: {offer.get('valid_until', 'See website')}"
            )
            metas.append({"source": "recommendations", "type": "seasonal_offer",
                          "offer_id": offer["offer_id"]})

        # Frequently bought together — one doc per primary product
        for fbt in data.get("frequently_bought_together", []):
            bought_with = "\n".join(
                f"{x['product_name']} ({int(x['match_rate'] * 100)}% of customers also buy this)"
                for x in fbt.get("bought_with", [])
            )
            ids.append(f"fbt_{fbt['primary_product']}")
            docs.append(f"Customers who buy {fbt['primary_name']} also frequently buy:\n{bought_with}")
            metas.append({"source": "recommendations", "type": "frequently_bought_together",
                          "primary_product": fbt["primary_product"]})

        # Personalised tips — one doc per tip
        for tip in data.get("personalised_tips", []):
            cat = tip["category"].replace(" ", "_").lower()
            ids.append(f"tip_{cat}")
            docs.append(f"Skincare/Beauty Tip — {tip['category']}:\n{tip['tip']}")
            metas.append({"source": "recommendations", "type": "beauty_tip"})

        return ids, docs, metas

    def _load_store_info(self):
        ids, docs, metas = [], [], []
        data = self._read_json("store_info.json")

        # Store branches — one doc per branch
        for branch in data.get("store_locations", []):
            bid      = branch["branch_id"]
            services = ", ".join(branch.get("services", []))
            ids.append(f"store_{bid}")
            docs.append(
                f"ShopEase Store: {branch['name']}\n"
                f"City: {branch['city']}, {branch['area']}\n"
                f"Address: {branch['address']}\n"
                f"Phone: {branch['phone']}\n"
                f"Hours: {branch['hours']}\n"
                f"Services: {services}\n"
                f"Parking: {branch.get('parking', 'See website')}"
            )
            metas.append({
                "source":    "store_info",
                "type":      "store_location",
                "city":      branch["city"],
                "area":      branch.get("area", ""),
                "branch_id": bid,
                "services":  services,
            })

        # Delivery — one doc per delivery type
        delivery = data.get("delivery", {})

        sd = delivery.get("same_day_delivery", {})
        ids.append("delivery_same_day")
        docs.append(
            f"Same-Day Delivery at ShopEase:\n"
            f"Available in: {', '.join(sd.get('available_cities', []))}.\n"
            f"Order before {sd.get('cutoff_time', '2 PM')}.\n"
            f"Fee: EGP {sd.get('fee', 49.99):.0f}."
        )
        metas.append({"source": "store_info", "type": "delivery", "delivery_type": "same_day"})

        nd = delivery.get("next_day_delivery", {})
        ids.append("delivery_next_day")
        docs.append(
            f"Next-Day Delivery at ShopEase:\n"
            f"Available in: {', '.join(nd.get('available_cities', []))}.\n"
            f"Fee: EGP {nd.get('fee', 29.99):.0f}."
        )
        metas.append({"source": "store_info", "type": "delivery", "delivery_type": "next_day"})

        std = delivery.get("standard_delivery", {})
        ids.append("delivery_standard")
        docs.append(
            f"Standard Delivery at ShopEase:\n"
            f"Coverage: {std.get('coverage', 'All governorates')}.\n"
            f"Delivery time: {std.get('days', '2-5 business days')}.\n"
            f"Fee: EGP {std.get('fee', 19.99):.0f}. Free above EGP {std.get('free_above', 500):.0f}."
        )
        metas.append({"source": "store_info", "type": "delivery", "delivery_type": "standard"})

        # Payment methods
        ids.append("payment_methods")
        docs.append("ShopEase Payment Methods:\n" + "\n".join(data.get("payment_methods", [])))
        metas.append({"source": "store_info", "type": "payment"})

        # Promotions — one doc per promo
        for promo in data.get("promotions", []):
            ids.append(f"promo_{promo['promo_id']}")
            docs.append(
                f"Promotion: {promo['name']}\n"
                f"Details: {promo['description']}\n"
                f"Promo code: {promo.get('promo_code', 'No code needed')}\n"
                f"Valid: {promo.get('valid_until', 'See website')}"
            )
            metas.append({"source": "store_info", "type": "promotion",
                          "promo_id": promo["promo_id"]})

        # Website help — one doc per topic
        for topic, content in data.get("website_help", {}).items():
            ids.append(f"webhelp_{topic}")
            docs.append(f"Website Help — {topic.replace('_', ' ').title()}:\n{content}")
            metas.append({"source": "store_info", "type": "website_help", "topic": topic})

        # Contact info
        contact = data.get("contact", {})
        ids.append("contact_info")
        docs.append(
            f"ShopEase Egypt Contact Information:\n"
            f"Phone: {contact.get('customer_support_phone', '19123')}\n"
            f"WhatsApp: {contact.get('whatsapp', '')}\n"
            f"Email: {contact.get('email', '')}\n"
            f"Support Hours: {contact.get('support_hours', '')}"
        )
        metas.append({"source": "store_info", "type": "contact"})

        return ids, docs, metas

    # ── public API ────────────────────────────────────────────────────────────

    def index_all(self) -> None:
        """Index all sources into ChromaDB. Safe to call multiple times (upserts)."""
        logger.info("Starting document indexing…")

        sources = [
            (COLLECTION_PRODUCTS,        self._load_products),
            (COLLECTION_FAQS,            self._load_faqs),
            (COLLECTION_POLICIES,        self._load_policies),
            (COLLECTION_MANUALS,         self._load_manuals),
            (COLLECTION_COSMETICS,       self._load_cosmetics),
            (COLLECTION_RECOMMENDATIONS, self._load_recommendations),
            (COLLECTION_STORE,           self._load_store_info),
        ]

        all_ids, all_docs, all_metas = [], [], []
        counts = {}

        for collection_name, loader in sources:
            ids, docs, metas = loader()
            self._upsert(self._collection(collection_name), ids, docs, metas)
            all_ids   += ids
            all_docs  += docs
            all_metas += metas
            counts[collection_name] = len(ids)

        self._upsert(self._collection(COLLECTION_ALL), all_ids, all_docs, all_metas)

        logger.info("Indexing complete. Total: %d documents.", len(all_ids))
        print(f"✓ Indexed {len(all_ids)} documents into ChromaDB.")
        for name, count in counts.items():
            print(f"  {name}: {count}")

    def get_collection_stats(self) -> dict:
        collections = [
            COLLECTION_PRODUCTS, COLLECTION_FAQS, COLLECTION_POLICIES,
            COLLECTION_MANUALS, COLLECTION_COSMETICS,
            COLLECTION_RECOMMENDATIONS, COLLECTION_STORE, COLLECTION_ALL,
        ]
        stats = {}
        for name in collections:
            try:
                stats[name] = self.client.get_collection(name).count()
            except Exception:
                stats[name] = 0
        return stats
