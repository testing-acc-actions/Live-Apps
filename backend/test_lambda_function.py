
import json
import unittest

from lambda_function import lambda_handler, normalize_comparison


def make_event(product_urls, priority="value_for_money", budget=30000):
    return {
        "productUrls": product_urls,
        "budget": budget,
        "priority": priority
    }


class LambdaHandlerTests(unittest.TestCase):

    def test_accepts_amazon_and_flipkart_urls(self):
        event = make_event([
            "https://www.amazon.in/dp/B0TEST12345",
            "https://www.flipkart.com/example-product/p/itm123456789"
        ])

        result = lambda_handler(event, None)
        body = json.loads(result["body"])

        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(body["status"], "validation_success")
        self.assertEqual(body["totalProducts"], 2)

    def test_rejects_less_than_two_products(self):
        event = make_event([
            "https://www.amazon.in/dp/B0TEST12345"
        ])

        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 400)

    def test_rejects_http_urls(self):
        event = make_event([
            "https://www.amazon.in/dp/B0TEST12345",
            "http://www.flipkart.com/example-product"
        ])

        result = lambda_handler(event, None)
        body = json.loads(result["body"])

        self.assertEqual(result["statusCode"], 400)
        self.assertEqual(body["status"], "correction_required")

    def test_rejects_unsupported_marketplace(self):
        event = make_event([
            "https://www.amazon.in/dp/B0TEST12345",
            "https://www.example.com/product"
        ])

        result = lambda_handler(event, None)
        body = json.loads(result["body"])

        self.assertEqual(result["statusCode"], 400)
        self.assertEqual(body["status"], "correction_required")

    def test_rejects_invalid_priority(self):
        event = make_event([
            "https://www.amazon.in/dp/B0TEST12345",
            "https://www.flipkart.com/example-product"
        ], priority="unknown")

        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 400)
        
    def test_rejects_zero_budget(self):
        event = make_event([
            "https://www.amazon.in/dp/B0TEST12345",
            "https://www.flipkart.com/example-product"
        ], budget=0)

        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 400)


    def test_ai_summary_rejects_incomplete_product_list(self):
        event = {
            "action": "ai_summary",
            "category": "smartphones",
            "budget": 90000,
            "priority": "lowest_price",
            "products": [
                {
                    "name": "Sample Phone A",
                    "price": 45000,
                    "features": {}
                }
            ]
        }

        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 400)


    def test_public_function_url_ai_summary_is_no_longer_blocked(self):
        event = {
            "action": "ai_summary",
            "requestContext": {
                "http": {
                    "method": "POST"
                }
            }
        }

        result = lambda_handler(event, None)

        # Missing category/productUrls should fail validation, not public-access blocking.
        self.assertEqual(result["statusCode"], 400)
        self.assertEqual(json.loads(result["body"])["status"], "validation_error")

    def test_rejects_oversized_public_request_body(self):
        event = {
            "body": "x" * 8193,
            "requestContext": {
                "http": {
                    "method": "POST"
                }
            }
        }

        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 413)

    def test_rejects_non_post_public_method(self):
        event = {
            "requestContext": {
                "http": {
                    "method": "GET"
                }
            }
        }

        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 405)


    def test_script_payload_is_classified_as_failed_retrieval(self):
        comparison = {
            "products": [{
                "product_name": "window.__NEXT_DATA__",
                "brand": "Unavailable",
                "key_specifications": [{
                    "name": "webpack chunk",
                    "value": "javascript application/ld+json"
                }]
            }],
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/example/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted)

        self.assertEqual(result["products"][0]["retrieval_status"], "failed")
        self.assertEqual(result["products"][0]["key_specifications"], [])
        self.assertEqual(result["products"][0]["price"]["amount"], None)
        self.assertEqual(result["recommendations"]["best_overall"], "insufficient data")

    def test_technical_payload_notes_override_ai_inferred_product_details(self):
        comparison = {
            "products": [
                {
                    "retrieval_status": "partial",
                    "retrieval_notes": "Page content consists of JavaScript bundles and New Relic monitoring payloads, not product listing information.",
                    "product_name": "Google Pixel 11 (256 GB Storage, 12 GB RAM)",
                    "brand": "Google",
                    "variant_match": True,
                    "price": {"amount": 79999, "status": "verified"},
                    "key_specifications": [
                        {"name": "Storage", "value": "256 GB"},
                        {"name": "RAM", "value": "12 GB"}
                    ]
                },
                {
                    "retrieval_status": "failed",
                    "retrieval_notes": "The URL could not be browsed.",
                    "product_name": "Motorola G37 Power",
                    "brand": "Motorola",
                    "key_specifications": []
                }
            ],
            "comparison": {},
            "recommendations": {},
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/pixel/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.flipkart.com/motorola/p/itm456", "marketplace": "Flipkart"}
        ]

        result = normalize_comparison(comparison, submitted)

        first = result["products"][0]
        self.assertEqual(first["retrieval_status"], "failed")
        self.assertEqual(first["product_name"], "Unavailable")
        self.assertEqual(first["brand"], "Unavailable")
        self.assertEqual(first["key_specifications"], [])
        self.assertIsNone(first["price"]["amount"])
        self.assertEqual(result["recommendations"]["best_overall"], "insufficient data")

    def test_two_products_with_supported_specs_can_be_verified(self):
        comparison = {
            "products": [
                {
                    "product_name": "Phone A",
                    "brand": "Brand A",
                    "identity_evidence": "Brand A Phone A product listing",
                    "variant_match": True,
                    "key_specifications": [
                        {"name": "Storage", "value": "256 GB", "evidence": "Storage capacity 256 GB"},
                        {"name": "Display", "value": "6.7 inch", "evidence": "Display size 6.7 inch"}
                    ]
                },
                {
                    "product_name": "Phone B",
                    "brand": "Brand B",
                    "identity_evidence": "Brand B Phone B product listing",
                    "variant_match": True,
                    "key_specifications": [
                        {"name": "Storage", "value": "128 GB", "evidence": "Storage capacity 128 GB"},
                        {"name": "Display", "value": "6.5 inch", "evidence": "Display size 6.5 inch"}
                    ]
                }
            ],
            "comparison": {},
            "recommendations": {},
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/a/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted)

        self.assertEqual(result["products"][0]["retrieval_status"], "verified")
        self.assertEqual(result["products"][1]["retrieval_status"], "verified")

    def test_only_one_usable_product_forces_insufficient_recommendations(self):
        comparison = {
            "products": [
                {
                    "product_name": "Phone A",
                    "brand": "Brand A",
                    "identity_evidence": "Brand A Phone A product listing",
                    "variant_match": True,
                    "key_specifications": [
                        {"name": "Storage", "value": "256 GB", "evidence": "Storage capacity 256 GB"},
                        {"name": "Display", "value": "6.7 inch", "evidence": "Display size 6.7 inch"}
                    ]
                },
                {
                    "product_name": "Unavailable",
                    "brand": "Unavailable",
                    "key_specifications": []
                }
            ],
            "comparison": {},
            "recommendations": {
                "best_overall": "Phone A",
                "best_value": "Phone A",
                "best_features": "Phone A"
            },
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/a/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted)

        self.assertEqual(result["products"][0]["retrieval_status"], "verified")
        self.assertEqual(result["products"][1]["retrieval_status"], "failed")
        self.assertEqual(result["recommendations"]["best_overall"], "insufficient data")
        self.assertEqual(result["comparison"]["feature_comparison"], [])

    def test_unsubstantiated_specs_and_prices_are_removed(self):
        comparison = {
            "products": [{
                "product_name": "Phone X",
                "brand": "Brand X",
                "identity_evidence": "Brand X Phone X product listing",
                "variant_match": True,
                "price": {"amount": 49999, "status": "verified"},
                "key_specifications": [
                    {"name": "RAM", "value": "16 GB"},
                    {"name": "Storage", "value": "1 TB", "evidence": "Storage capacity 1 TB"}
                ]
            }],
            "comparison": {},
            "recommendations": {},
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/x/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]
        result = normalize_comparison(comparison, submitted)
        product = result["products"][0]
        self.assertEqual(len(product["key_specifications"]), 1)
        self.assertEqual(product["key_specifications"][0]["name"], "Storage")
        self.assertIsNone(product["price"]["amount"])
        self.assertEqual(product["price"]["status"], "not_found")
        self.assertEqual(product["retrieval_status"], "partial")

    def test_price_requires_page_evidence(self):
        comparison = {
            "products": [{
                "product_name": "Phone X",
                "brand": "Brand X",
                "identity_evidence": "Brand X Phone X product listing",
                "variant_match": True,
                "price": {
                    "amount": 49999,
                    "status": "verified",
                    "evidence": "Special price Rs. 49,999"
                },
                "key_specifications": []
            }],
            "comparison": {},
            "recommendations": {},
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/x/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]
        result = normalize_comparison(comparison, submitted)
        self.assertEqual(result["products"][0]["price"]["amount"], 49999)
        self.assertEqual(result["products"][0]["price"]["evidence"], "Special price Rs. 49,999")



    def test_price_is_recovered_when_amount_is_null_but_page_quote_is_valid(self):
        comparison = {
            "products": [{
                "product_name": "Phone X",
                "brand": "Brand X",
                "identity_evidence": "Brand X Phone X product listing",
                "variant_match": True,
                "price": {
                    "amount": None,
                    "status": "not_found",
                    "evidence": "Special price ₹89,999"
                },
                "key_specifications": []
            }],
            "comparison": {},
            "recommendations": {},
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/x/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted, budget=70000)
        product = result["products"][0]

        self.assertEqual(product["price"]["amount"], 89999)
        self.assertEqual(product["price"]["status"], "verified")
        self.assertEqual(product["budget_status"], "over_budget")

    def test_price_is_rejected_when_amount_conflicts_with_page_quote(self):
        comparison = {
            "products": [{
                "product_name": "Phone X",
                "brand": "Brand X",
                "identity_evidence": "Brand X Phone X product listing",
                "variant_match": True,
                "price": {
                    "amount": 49999,
                    "status": "verified",
                    "evidence": "Special price ₹89,999"
                },
                "key_specifications": []
            }],
            "comparison": {},
            "recommendations": {},
            "data_limitations": []
        }
        submitted = [
            {"url": "https://www.flipkart.com/x/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted)
        product = result["products"][0]

        self.assertIsNone(product["price"]["amount"])
        self.assertEqual(product["price"]["status"], "not_found")



    def test_partial_variant_is_never_recommended(self):
        comparison = {
            "products": [
                {
                    "product_name": "Phone Exact",
                    "brand": "Brand",
                    "identity_evidence": "Brand Phone Exact product listing",
                    "variant_match": True,
                    "key_specifications": [
                        {"name": "RAM", "value": "8 GB", "evidence": "8 GB RAM"},
                        {"name": "Storage", "value": "128 GB", "evidence": "128 GB storage"}
                    ]
                },
                {
                    "product_name": "Phone Wrong Colour",
                    "brand": "Brand",
                    "identity_evidence": "Brand Phone Wrong Colour product listing",
                    "model_match": True,
                    "storage_match": True,
                    "color_match": False,
                    "key_specifications": [
                        {"name": "RAM", "value": "8 GB", "evidence": "8 GB RAM"},
                        {"name": "Storage", "value": "128 GB", "evidence": "128 GB storage"}
                    ]
                }
            ],
            "comparison": {},
            "recommendations": {
                "best_overall": "Phone Wrong Colour",
                "best_value": "Phone Wrong Colour",
                "best_features": "Phone Wrong Colour"
            }
        }
        submitted = [
            {"url": "https://www.flipkart.com/exact/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.flipkart.com/partial/p/itm456", "marketplace": "Flipkart"}
        ]

        result = normalize_comparison(comparison, submitted, budget=30000)

        self.assertEqual(result["products"][1]["variant_status"], "partial")
        self.assertEqual(result["recommendations"]["best_overall"], "insufficient data")
        self.assertEqual(result["recommendations"]["best_value"], "insufficient data")
        self.assertEqual(result["recommendations"]["best_features"], "insufficient data")

    def test_best_value_rejects_over_budget_product(self):
        comparison = {
            "products": [
                {
                    "product_name": "Affordable Phone",
                    "brand": "Brand",
                    "identity_evidence": "Brand Affordable Phone product listing",
                    "variant_match": True,
                    "price": {
                        "amount": 25000,
                        "evidence": "Special price ₹25,000"
                    },
                    "key_specifications": [
                        {"name": "RAM", "value": "8 GB", "evidence": "8 GB RAM"},
                        {"name": "Storage", "value": "128 GB", "evidence": "128 GB storage"}
                    ]
                },
                {
                    "product_name": "Expensive Phone",
                    "brand": "Brand",
                    "identity_evidence": "Brand Expensive Phone product listing",
                    "variant_match": True,
                    "price": {
                        "amount": 90000,
                        "evidence": "Special price ₹90,000"
                    },
                    "key_specifications": [
                        {"name": "RAM", "value": "12 GB", "evidence": "12 GB RAM"},
                        {"name": "Storage", "value": "256 GB", "evidence": "256 GB storage"}
                    ]
                }
            ],
            "comparison": {},
            "recommendations": {
                "best_overall": "Expensive Phone",
                "best_value": "Expensive Phone",
                "best_features": "Expensive Phone"
            }
        }
        submitted = [
            {"url": "https://www.flipkart.com/affordable/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0EXPENSIVE1", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted, budget=70000)

        self.assertEqual(result["recommendations"]["best_overall"], "Expensive Phone")
        self.assertEqual(result["recommendations"]["best_value"], "insufficient data")
        self.assertEqual(result["recommendations"]["best_features"], "Expensive Phone")

    def test_best_value_can_only_select_an_exact_verified_within_budget_product(self):
        comparison = {
            "products": [
                {
                    "product_name": "Affordable Phone",
                    "brand": "Brand",
                    "identity_evidence": "Brand Affordable Phone product listing",
                    "variant_match": True,
                    "price": {
                        "amount": 25000,
                        "evidence": "Special price ₹25,000"
                    },
                    "key_specifications": [
                        {"name": "RAM", "value": "8 GB", "evidence": "8 GB RAM"},
                        {"name": "Storage", "value": "128 GB", "evidence": "128 GB storage"}
                    ]
                },
                {
                    "product_name": "Expensive Phone",
                    "brand": "Brand",
                    "identity_evidence": "Brand Expensive Phone product listing",
                    "variant_match": True,
                    "price": {
                        "amount": 90000,
                        "evidence": "Special price ₹90,000"
                    },
                    "key_specifications": [
                        {"name": "RAM", "value": "12 GB", "evidence": "12 GB RAM"},
                        {"name": "Storage", "value": "256 GB", "evidence": "256 GB storage"}
                    ]
                }
            ],
            "comparison": {},
            "recommendations": {
                "best_overall": "Affordable Phone",
                "best_value": "Affordable Phone",
                "best_features": "Expensive Phone"
            }
        }
        submitted = [
            {"url": "https://www.flipkart.com/affordable/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0EXPENSIVE1", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted, budget=70000)

        self.assertEqual(result["recommendations"]["best_value"], "Affordable Phone")

    def test_strengths_and_tradeoffs_require_page_evidence(self):
        comparison = {
            "products": [
                {
                    "product_name": "Phone X",
                    "brand": "Brand X",
                    "identity_evidence": "Brand X Phone X product listing",
                    "variant_match": True,
                    "strengths": [
                        {"text": "Large battery", "evidence": "5000 mAh battery capacity"},
                        {"text": "Unsupported fast charging claim"}
                    ],
                    "tradeoffs": [
                        {"text": "Heavy device", "evidence": "Weight: 220 g"}
                    ],
                    "key_specifications": [
                        {"name": "Battery", "value": "5000 mAh", "evidence": "5000 mAh battery capacity"},
                        {"name": "Weight", "value": "220 g", "evidence": "Weight: 220 g"}
                    ]
                }
            ],
            "comparison": {},
            "recommendations": {}
        }
        submitted = [
            {"url": "https://www.flipkart.com/phone-x/p/itm123", "marketplace": "Flipkart"},
            {"url": "https://www.amazon.in/dp/B0TEST12345", "marketplace": "Amazon"}
        ]

        result = normalize_comparison(comparison, submitted)
        product = result["products"][0]

        self.assertEqual(product["strengths"], ["Large battery"])
        self.assertEqual(product["tradeoffs"], ["Heavy device"])

if __name__ == "__main__":
    unittest.main()
