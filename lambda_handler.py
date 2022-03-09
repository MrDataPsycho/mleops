import json
from inference_onnx import ColaONNXPredictor

predictor = ColaONNXPredictor("./models/model.onnx")


def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of the given sentence
    :param event: post/get request context
    :param context: Context of Meta data
    """

    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        # print(f"Got the input: {body['sentence']}")
        response = predictor.predict(body["sentence"])
        return {
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(response)
        }
    else:
        return predictor.predict(event["sentence"])


if __name__ == "__main__":
    test = {"sentence": "this is a sample sentence"}
    lambda_handler(test, None)
