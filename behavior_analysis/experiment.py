import h5py
import numpy as np
import pandas as pd
from exptlib import Experiment, MetadataAttribute, CSVMetadata
import cv2
from exptlib.directory import Directory
from sigseg import align_peaks, find_runs
from scipy.interpolate import CubicSpline

from .io import TrackingInterface
from .kinematics import compute_tail_curvature, tail_angle_filter, find_bouts, interpolate_frame_rate


class PreyCaptureExperiment(Experiment):

    species_names = {
        "d_rerio": "Danio rerio",
    }

    video_data = MetadataAttribute(CSVMetadata, "video_data.csv", write_kw=dict(index=False))

    def create_metadata(self):
        videos = []
        for species, species_directory in self.data_directory.items():
            print(species)
            for date, date_directory in species_directory["videos"].items():
                print("\t" + date)
                for fish, fish_directory in date_directory.items():
                    print("\t\t" + fish)
                    fish_id = self.fish_id(species, date, fish)
                    for path in fish_directory.glob("*.avi"):
                        timestamp = path.stem
                        video_id = "_".join([fish_id, timestamp.replace("-", "")])
                        print("\t\t\t" + video_id)
                        stats = self.video_statistics(path)
                        videos.append([self.species_names[species],
                                       species,
                                       date,
                                       fish,
                                       fish_id,
                                       timestamp,
                                       video_id,
                                       stats["frame_rate"],
                                       stats["n_frames"],
                                       stats["width"],
                                       stats["height"]])
        videos = pd.DataFrame(videos, columns=["species_name",
                                               "species_id",
                                               "date",
                                               "fish_name",
                                               "fish_id",
                                               "timestamp",
                                               "video_id",
                                               "frame_rate",
                                               "n_frames",
                                               "width",
                                               "height"])
        self.video_data = videos

    def get(self, key, val):
        return self.video_data.groupby(key).get_group(val)

    @staticmethod
    def video_statistics(path):
        cap = cv2.VideoCapture(str(path))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        return {"frame_rate": frame_rate,
                "n_frames": int(n_frames),
                "width": int(w), "height": int(h)}

    @staticmethod
    def fish_id(species, date, fish_name):
        genus, species = species.upper().split("_")
        date = date.replace("_", "")
        fish_idx = "".join([c for c in fish_name if c.isdigit()])
        fish_idx = fish_idx.rjust(2, "0")
        return "".join([genus[0], species[0], date, fish_idx])


class cached:

    def __init__(self, method):
        self.method = method

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

    def __get__(self, instance, owner):
        try:
            val = getattr(instance, self.private_name)
        except AttributeError:
            val = self.method(instance)
            setattr(instance, self.private_name, val)
        return val


class TrialData:

    directory = Directory("")
    eye_directory = Directory("")

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.transformed = None
        self.filtered = None

    def __getitem__(self, item):
        return getattr(self, item)

    # eye angles
    # prey positions
    @property
    def eye_tracking_directory(self):
        return self.eye_directory[self["species_id"], self["fish_id"]]

    @property
    def eye_tracking_path(self):
        return self.eye_tracking_directory[self["video_id"] + ".h5"]

    def eye_points(self):
        path = self.eye_tracking_path
        with h5py.File(path, "r") as f:
            tracking = f["tracks"][0][:]
        return tracking

    def eye_angles(self):
        points = self.eye_points()
        left = points[:, (0, 2)]
        right = points[:, (1, 3)]
        sb_mid = points[:, (6, 5, 4)]
        # Vectors
        sb_mid = sb_mid[:, 0] - sb_mid[:, 1:].mean(axis=1)
        left = np.diff(left, axis=1).squeeze()
        right = np.diff(right, axis=1).squeeze()
        # Transpose
        sb_mid = sb_mid.T
        left = left.T
        right = right.T
        # Normalize
        sb_mid = sb_mid / np.linalg.norm(sb_mid, axis=1, keepdims=True)
        left = left / np.linalg.norm(left, axis=1, keepdims=True)
        right = right / np.linalg.norm(right, axis=1, keepdims=True)
        # Angles
        left_sign = np.sign(np.cross(sb_mid, left))
        left_angle = np.arccos(np.einsum("ni,ni->n", sb_mid, left)) * left_sign
        right_sign = np.sign(np.cross(sb_mid, right))
        right_angle = np.arccos(np.einsum("ni,ni->n", sb_mid, right)) * right_sign
        angles = np.array([left_angle, right_angle]).T
        return angles

    @property
    def tracking_directory(self):
        return self.directory[self["species_id"], self["fish_id"]]

    @property
    def tracking_path(self):
        return self.tracking_directory[self["video_id"] + ".h5"]

    @property
    def tracking(self):
        return TrackingInterface(self.tracking_path)

    @cached
    def tail_curvature(self):
        points = self.tracking.tail_points()
        headings = self.tracking.headings()
        ks, _ = compute_tail_curvature(points, headings)
        return ks

    def transform_tail(self, components, mean, std):
        ks = self.tail_curvature  # get tail curvature
        transformed = np.dot((ks - mean) / std, components.T)  # whiten and project
        self.transformed = transformed  # set attribute
        return transformed

    def interpolate(self, gap_size=4):
        # get transformed
        x = self.transformed
        # find gaps to interpolate
        x_nan = np.any(np.isnan(x), axis=1)
        gaps, = np.where(x_nan)
        gaps = find_runs(gaps)
        gaps = [idxs for idxs in gaps if len(idxs) <= gap_size]
        if len(gaps):
            # get indices to interpolate
            interp_t = [idxs for idxs in gaps if
                        (~np.any(x_nan[idxs[0] - 3:idxs[0]]) & ~np.any(x_nan[idxs[-1] + 1:idxs[-1] + 4]))]
            if not len(interp_t):
                return self.transformed
            interp_t = np.hstack(interp_t)
            # interpolate
            t, = np.where(~x_nan)
            cs = CubicSpline(t, x[t])
            interp_x = cs(interp_t)
            self.transformed[interp_t] = interp_x
        return self.transformed

    def interpolate_frame_rate(self, fps_new) -> (np.ndarray, np.ndarray):
        x = self.transformed
        fps_old = self.frame_rate
        return interpolate_frame_rate(x, fps_old, fps_new)

    @cached
    def distance_transformed(self):
        if self.transformed is None:
            raise ValueError("Must map tail onto principal components (transform_tail).")
        return np.linalg.norm(self.transformed, axis=1)

    def filter_tail(self, winsize):
        self.filtered = tail_angle_filter(self.distance_transformed, winsize)
        return self.filtered

    def find_bouts(self, threshold, minsize, winsize=None, velocity_threshold=0.):
        x = self.filtered
        if x is None:
            if winsize is None:
                raise ValueError("Must specify winsize if trial is not already filtered.")
            x = self.filter_tail(winsize)
        events = find_bouts(x, threshold, minsize=minsize)
        velocity = self.tracking.velocity()
        bouts = []
        for (start, peak, end) in events:
            if start == 0:
                continue
            if (start > 0) and np.isnan(x[start - 1]):
                continue
            if end == len(x):
                continue
            if np.isnan(x[end]):
                continue
            maxvel = np.max(velocity[start:end])
            if maxvel > velocity_threshold:
                bouts.append((self.video_id, start, peak, end))
        bouts = pd.DataFrame(bouts, columns=["trial", "start", "peak", "end"])
        return bouts

    @staticmethod
    def from_bouts(x: np.ndarray, bouts: pd.DataFrame):
        bout_traces = []
        for idx, bout_info in bouts.iterrows():
            bout_traces.append(x[bout_info.start:bout_info.end])
        return bout_traces

    def align(self, bouts: pd.DataFrame, before: int, after: int):
        xs = self.from_bouts(self.transformed, bouts)
        peaks = (bouts["peak"] - bouts["start"]).values
        xs_aligned = align_peaks(xs, peaks, before, after)
        return xs_aligned

    def seconds_to_frames(self, t):
        if isinstance(t, (int, float)):
            return int(np.round(t * self.frame_rate))
        return np.round(np.array(t) * self.frame_rate).astype("i4")

    def frames_to_seconds(self, frames):
        return np.array(frames) / self.frame_rate
