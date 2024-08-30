import click
import time
import itertools
from pathlib import Path

import structlog
import cv2 as cv
import numpy as np

log = structlog.get_logger()


"""
TODO:

[x] script, offline
[x] output coordinates of objects in each frame
[x] visualization tracked path
[x] opencv
[x] detect circles and rectangles
[x] object entity persistence - 2.5% loss of visibility
[x] final visual report of each object path
[/] tool simple to understand
[/] easily identifiable mistakes
[x] basic docs, object detector, tracker, app
[x] code quality
[/] efficiency
[/] accuracy
[/] creativity/effectivenes
[x] zip

"""


class Entity:
    """
    Entity to be visulaly tracked with persistance in case of short loss of acquisition.

    TODO: Use Kalman filter for localization and velocity vector filtering (maybe)
    """

    default_ttl = 5
    id_ctr = itertools.count()

    def __init__(self, color: np._typing.ArrayLike, shape_class: str, center):
        self.color = color
        self.shape_class = shape_class
        self.track = list()
        self.track.append(center)
        self.ttl = self.default_ttl
        self.id = next(self.id_ctr)
        self.dead = False

    def __repr__(self):
        return f"E(id={self.id}, ttl={self.ttl})"

    @property
    def curr_pos(self):
        return self.track[-1]

    def velocity(self, frame_window=10):
        track_window = self.track[-frame_window:]
        return (track_window[-1] - track_window[0]) / frame_window

    def update_track(self, center, imputed: bool):
        """
        Record a new location.
        """
        self.track.append(center)

        if imputed is False:
            self.ttl = self.default_ttl

    def like_shape(self, shape, dist_tol=20, color_tol=20) -> bool:
        """
        Is Entity similar enough to be seen as this shape.
        """
        if cv.norm(shape["color"], self.color) > color_tol:
            return False
        if (shape["shape_class"] is not None) and (
            shape["shape_class"] != self.shape_class
        ):
            return False
        if cv.norm(shape["center"], self.curr_pos) > dist_tol:
            return False

        return True


class ShapeTracker(object):
    """Tracks circles and rectangles in a video"""

    def __init__(self, video_path: Path):
        assert Path(video_path).exists()

        self.cap = cv.VideoCapture(str(video_path))
        self.log = log.bind()
        self.entities = []
        self.frame_shape = None

    def run(self, delay=0, limit=0):
        frame_num = 0

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if self.frame_shape is None:
                    self.frame_shape = frame.shape

                frame = self.process_frame(frame, frame_num)

                if not ret:
                    log.info("End of stream")
                    break

                cv.imshow("frame", frame)
                if cv.waitKey(1) == ord("q"):
                    break

                frame_num += 1

                if limit > 0 and frame_num > limit:
                    break

                time.sleep(delay)

        finally:
            self.cap.release()
            cv.destroyAllWindows()

    @staticmethod
    def get_colors(frame):
        """
        Get dominant colors.

        TODO:
            - Picking colors is ULTRA slow - a tradeoff for being able to distinguish coliding colors
            - Maybe use color clustering
        """
        frame = frame[::4, ::4]

        # TODO slow
        colors, counts = np.unique(
            frame.reshape(-1, frame.shape[-1]), axis=0, return_counts=True
        )
        bright_mask = np.sum(colors, axis=1) > 20
        popular_mask = counts > 20
        return colors[bright_mask & popular_mask]

    def get_shapes(
        self, frame, clr_tolerance=20, min_area=20, max_area=10e4
    ) -> list[dict]:
        """
        Based on color palette, extract binary masks and extract top-most shapes/contours from them.
        """

        colors = self.get_colors(frame)
        shapes = []

        frame = cv.blur(frame, [8, 8])
        for color in colors:
            c_upper = color + clr_tolerance  # overflows
            c_upper[c_upper < color] = 255
            c_lower = color - clr_tolerance  # underflows
            c_lower[c_lower > color] = 0
            mask = cv.inRange(frame, lowerb=c_lower, upperb=c_upper)

            contours, _hierarchy = cv.findContours(
                image=mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE
            )

            for cont in contours:
                if cv.contourArea(cont) < min_area:
                    continue
                if cv.contourArea(cont) > max_area:
                    continue

                br = np.array(cv.boundingRect(cont))
                if self.is_rect(cont):
                    shapes_class = "rect"
                elif self.is_circle(cont):
                    shapes_class = "circ"
                else:
                    shapes_class = None

                shape = {
                    "shape_class": shapes_class,
                    "bounding_rect": br,
                    "center": (br[:2] + br[2:] / 2).astype(int),
                }
                shape["color"] = frame[*shape["center"][::-1]].copy()

                shapes.append(shape)

        return shapes

    @staticmethod
    def is_rect(shape):
        _x, _y, w, h = cv.boundingRect(shape)
        bounding_area = w * h
        contour_area = cv.contourArea(shape)
        return contour_area / bounding_area > 0.9

    @staticmethod
    def is_circle(shape):
        x, y, w, h = cv.boundingRect(shape)
        center = np.array([x + w / 2, y + h / 2])
        radii = [cv.norm(center - p) for p in shape]
        # TODO maybe check angle distribution
        return np.std(radii) < 3

    @staticmethod
    def draw_shapes(frame, shapes):
        for shape in shapes:
            br = shape["bounding_rect"]
            cv.rectangle(
                img=frame,
                pt1=br[:2],
                pt2=br[:2] + br[2:],
                color=(255, 0, 0),
                thickness=3,
            )
            cv.circle(
                img=frame, center=shape["center"].tolist(), radius=1, color=(255, 0, 0)
            )

        return frame

    @staticmethod
    def draw_entities(frame, entities, vector_scale=10, trail=True):
        for e in entities:
            frame = cv.circle(
                img=frame, center=e.curr_pos, radius=1, color=(66, 66, 66), thickness=3
            )
            frame = cv.putText(
                img=frame,
                text=f"{e.id}:ttl={e.ttl},pos={e.curr_pos[0].item(),e.curr_pos[1].item()}",
                org=e.curr_pos,
                fontFace=cv.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(255, 255, 255),
            )
            frame = cv.line(
                img=frame,
                pt1=e.curr_pos,
                pt2=(e.curr_pos + e.velocity() * vector_scale).astype(int),
                color=(0, 222, 222),
                thickness=3,
            )
            if trail and len(e.track) > 1:
                track = np.concatenate(e.track).reshape(-1, 2)
                cv.polylines(
                    img=frame,
                    pts=[track],
                    isClosed=False,
                    color=(22, 22, 22),
                    thickness=1,
                )

        return frame

    def process_frame(self, frame, frame_num):
        start_t = time.time()
        self.log.bind(frame_num=frame_num)

        shapes = self.get_shapes(frame)
        # frame = self.draw_shapes(frame, shapes)

        for e in self.entities:
            e.ttl -= 1

        # Assign shapes to entities
        while len(shapes):
            shape = shapes.pop()

            for entity in [e for e in self.entities if not e.dead]:
                if entity.like_shape(shape):
                    if entity.ttl != entity.default_ttl:
                        entity.update_track(
                            (shape["center"]).astype(int), imputed=False
                        )
                    break
            else:
                self.entities.append(
                    Entity(
                        color=shape["color"],
                        shape_class=shape["shape_class"],
                        center=shape["center"],
                    )
                )

        # Extrapolate position on lost entities
        for e in self.entities:
            if e.ttl < e.default_ttl:
                imputed_center = (e.curr_pos + e.velocity()).astype(int)
                e.update_track(center=imputed_center, imputed=True)

        for e in self.entities:
            if e.ttl <= 0:
                e.dead = True

        frame = self.draw_entities(frame, [e for e in self.entities if not e.dead])

        self.log.debug(
            "Frame processed",
            frame_num=frame_num,
            entities=len(self.entities),
            ms=round((time.time() - start_t) * 1000, 0),
        )

        return frame

    def save_reports(self):
        for e in self.entities:
            frame = np.zeros(self.frame_shape)
            frame = self.draw_entities(frame, [e])
            cv.imwrite(f"report_entity_{e.id}.jpg", frame)


@click.command()
@click.option(
    "--reports", is_flag=True, help="Save report images for each entity tracker"
)
@click.option(
    "--delay", type=float, help="Delay each preview frame by <float> seconds", default=0
)
@click.option(
    "--limit", type=int, help="Process only first <int> of frames.", default=0
)
@click.option(
    "--video-path",
    type=click.Path(exists=True),
    help="Process <filename> video file.",
    required=True,
)
def main(video_path, reports, delay, limit):
    shape_tracker = ShapeTracker(video_path=video_path)
    shape_tracker.run(delay=delay, limit=limit)
    if reports:
        shape_tracker.save_reports()


if __name__ == "__main__":
    main()
