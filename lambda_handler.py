import json
from src.factory.onnx import ColaONNXPredictor
from src.pathconfig import PathConfig

PATHFINDER = PathConfig()
MODEL = PATHFINDER.models.joinpath("model.onnx")
predictor = ColaONNXPredictor(str(MODEL))


def handler(event, context):
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
    result = handler(test, None)
    print(result)
