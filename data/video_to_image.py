"""Data recipe to transform input video to images"""

import cv2
import os
import shutil
import numpy as np
import pandas as pd
from h2oaicore.data import CustomData

_global_modules_needed_by_name = ["torchvision==0.4.1", "facenet-pytorch==2.2.9"]
from facenet_pytorch import MTCNN


class VideoToFrames:
    """
    Transforms input video files into image frames.
    Additionally detects all faces for each frame.
    """

    def __init__(self, num_frames_per_video=3, face_additional_area=0.5):
        self.num_frames_per_video = num_frames_per_video
        self.face_additional_area = face_additional_area

        self.face_detection_model = MTCNN(
            image_size=224,
            margin=0,
            keep_all=True,
            select_largest=False,
            post_process=False,
            thresholds=[0.8, 0.9, 0.9],
            device="cuda",
        ).eval()

    def video_to_frames(self, video_path, output_path):
        output_image_paths = []
        video_id = os.path.split(video_path)[-1]

        # Read video
        orig_capture = cv2.VideoCapture(video_path)

        # Select only self.num_frames_per_video uniform frames
        n_frames = int(orig_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_idx = np.linspace(
            0, n_frames, self.num_frames_per_video, endpoint=False, dtype=np.int
        )

        # Loop through all frames
        for frame_num in range(n_frames):
            ret = orig_capture.grab()

            if not ret:
                continue

            # Retrieve only required frames
            if frame_num in frames_idx:
                ret, frame_orig = orig_capture.retrieve()

                if ret:
                    # Save the whole video frame to the image
                    # img_path = os.path.join(output_path, f"{video_id}_frame_{frame_num}.png")
                    # cv2.imwrite(frame_orig, img_path)
                    # output_image_paths.append(os.path.split(img_path)[-1])

                    # Skip the next part if want to save the whole frame only
                    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)

                    # Detect all faces
                    faces, _ = self.face_detection_model.detect(frame_orig)
                    if faces is None:
                        return []

                    # For each detected face
                    for face_id, box in enumerate(faces):

                        # Get face coordinates
                        c0_start, c0_end, c1_start, c1_end = self.get_face_coordinates(
                            frame_orig, box
                        )

                        # Crop face
                        face_full = frame_orig[c0_start:c0_end, c1_start:c1_end]

                        # Save face to the file
                        img_path = os.path.join(
                            output_path,
                            f"{video_id}_frame_{frame_num}_face_{face_id}.png",
                        )
                        # Return BGR before saving
                        face_full = cv2.cvtColor(face_full, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(img_path, face_full)
                        output_image_paths.append(os.path.split(img_path)[-1])

        return output_image_paths

    def get_face_coordinates(self, frame_orig, box):
        sh0_start = int(box[1])
        sh0_end = int(box[3])
        sh1_start = int(box[0])
        sh1_end = int(box[2])

        # Add area around the face
        d0 = int((sh0_end - sh0_start) * self.face_additional_area)
        d1 = int((sh1_end - sh1_start) * self.face_additional_area)

        c0_start = max(sh0_start - d0, 0)
        c0_end = min(sh0_end + d0, frame_orig.shape[0])

        c1_start = max(sh1_start - d1, 0)
        c1_end = min(sh1_end + d1, frame_orig.shape[1])

        return c0_start, c0_end, c1_start, c1_end


class VideoDataset(CustomData):
    """
    Takes input video files and re-writes them as frame images
    """

    @staticmethod
    def create_data(X=None):

        # Path to a .csv with labels. First column is path to the video, second column is label
        path_to_labels = "/path/to/labels.csv"
        files_dir = os.path.split(path_to_labels)[0]

        output_path = os.path.join(files_dir, "video_frames/")
        os.makedirs(output_path, exist_ok=True)

        # Read data
        df = pd.read_csv(path_to_labels)
        video2label = dict(
            zip(df.iloc[:, 0].map(lambda x: os.path.split(x)[-1]), df.iloc[:, 1])
        )

        # Convert video to image frames and save them
        vid2frames = VideoToFrames()

        video_faces = {}
        for path in df.iloc[:, 0]:
            image_paths = vid2frames.video_to_frames(
                video_path=path, output_path=output_path
            )

            # If there are available images
            if len(image_paths) > 0:
                video_faces[os.path.split(path)[-1]] = image_paths

        output_df = pd.DataFrame(
            [(key, var) for (key, L) in video_faces.items() for var in L],
            columns=["video_id", "image_id"],
        )
        output_df["label"] = output_df.video_id.map(video2label)

        # Save .csv file with labels
        output_df.to_csv(os.path.join(output_path, "labels.csv"), index=False)

        # Create .zip archive to upload to DAI
        shutil.make_archive(
            base_name=os.path.join(files_dir, "video_frames"),
            format="zip",
            root_dir=output_path,
            base_dir=output_path,
        )
        zip_archive_path = os.path.join(files_dir, "video_frames.zip")

        return zip_archive_path
