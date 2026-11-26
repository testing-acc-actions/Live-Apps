
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


if __name__ == "__main__":
    unittest.main()
