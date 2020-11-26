import edgeiq
import cv2


def main():
    # Change parameter to alwaysai/human-pose-eyecloud to run the human pose model
    with edgeiq.EyeCloud('alwaysai/mobilenet_ssd_eyecloud'
                         ) as camera, edgeiq.Streamer() as streamer:

        while True:
            text = []

            frame = camera.read()

            result = camera.get_model_results()

            # Check for inferencing results.
            if result:
                text.append("Model: {}".format(camera.model_id))

                if camera.model_purpose == 'PoseEstimation':
                    frame = result.draw_poses(frame)

                    text.append("Inference time: {:1.3f} s".format(
                        result.duration))

                    for ind, pose in enumerate(result.poses):
                        text.append("Person {}".format(ind))
                        text.append('-' * 10)
                        text.append("Key Points:")
                        for key_point in pose.key_points:
                            text.append(str(key_point))

                elif camera.model_purpose == 'ObjectDetection':
                    frame = edgeiq.markup_image(frame, result.predictions)

                    text.append("Inference time: {:1.3f} s".format(
                        result.duration))
                    text.append("Objects:")

                    for prediction in result.predictions:
                        text.append("{}: {:2.2f}%".format(
                            prediction.label, prediction.confidence * 100))

                elif camera.model_purpose == 'Classification':
                    if len(result.predictions) > 0:
                        top_prediction = result.predictions[0]
                        text = "Classification: {}, {:.2f}%".format(
                            top_prediction.label,
                            top_prediction.confidence * 100)
                    else:
                        text = None

                    cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 0, 255), 2)

            streamer.send_data(frame, text)

            if streamer.check_exit():
                break


if __name__ == '__main__':
    main()
