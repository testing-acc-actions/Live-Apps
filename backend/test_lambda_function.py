
import json
import unittest

from lambda_function import lambda_handler


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


    def test_public_function_url_cannot_call_ai_summary(self):
        event = {
            "action": "ai_summary",
            "requestContext": {
                "http": {
                    "method": "POST"
                }
            }
        }

        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 403)


if __name__ == "__main__":
    unittest.main()
