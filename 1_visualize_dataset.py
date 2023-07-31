# 0_visualize_dataset.py

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box

import numpy as np
import pyquaternion
from typing import List, Dict
import plotly.graph_objects as PlotlyGraphObjects
import pprint

NUSCENES_DATASET_LOC = '/home/cdahms/NuScenesDataset/mini'
TRIP_ID = 7   # can set this to any valid trip ID

def main():
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    # load dataset
    print('\n' + 'loading dataset . . . ' + '\n')
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DATASET_LOC)

    print('len(nusc.scene) = ' + str(len(nusc.scene)))

    frameIds = []
    for sceneInfo in nusc.scene:
        frameId = sceneInfo['first_sample_token']

        frameIds.append(frameId)

        # while frameId is not None and frameId != '':
        #     frameIds.append(frameId)
        #     sampleData: Dict = nusc.get('sample', frameId)
        #     frameId = sampleData['next']
        # # end while
    # end for

    for frameId in frameIds:
        # go from frame ID to lidar points
        sampleData: Dict = nusc.get('sample', frameId)
        lidarTopId: str = sampleData['data']['LIDAR_TOP']
        lidarFilePath: str = nusc.get_sample_data_path(lidarTopId)
        lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lidarFilePath)
        lidarPoints: np.ndarray = lidarPointCloud.points

        print('\n' + 'lidarPoints: ')
        print(lidarPoints.shape)
        print('')

        # for lidar points, first 3 rows are x, y, z, 4th row is intensity which we don't need so remove it
        lidarPoints: np.ndarray = lidarPoints[:3, :]

        lidarTopData: dict = nusc.get('sample_data', lidarTopId)
        gndTrBoxes: List[Box] = nusc.get_boxes(lidarTopId)

        ### 3D visualization ######################################################

        s3dPoints = PlotlyGraphObjects.Scatter3d(x=lidarPoints[0], y=lidarPoints[1], z=lidarPoints[2], mode='markers', marker={'size': 1})

        # 3 separate lists for the x, y, and z components of each line
        xLines = []
        yLines = []
        zLines = []
        for box in gndTrBoxes:

            box = moveBoxFromWorldSpaceToSensorSpace(nusc, box, lidarTopData)

            corners = box.corners()

            # see here for documentation of Box:
            # https://github.com/nutonomy/nuscenes-devkit/blob/c44366daea8bba29673943c1fc86d0bfbfb7a99e/python-sdk/nuscenes/utils/data_classes.py#L521
            # when getting corners, the first 4 corners are the ones facing forward, the last 4 are the ones facing rearwards

            corners = corners.transpose()

            # 4 lines for front surface of box
            addLineToPlotlyLines(corners[0], corners[1], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[1], corners[2], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[2], corners[3], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[3], corners[0], xLines, yLines, zLines)

            # 4 lines between front points and read points
            addLineToPlotlyLines(corners[0], corners[4], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[1], corners[5], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[2], corners[6], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[3], corners[7], xLines, yLines, zLines)

            # 4 lines for rear surface of box
            addLineToPlotlyLines(corners[4], corners[7], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[5], corners[4], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[6], corners[5], xLines, yLines, zLines)
            addLineToPlotlyLines(corners[7], corners[6], xLines, yLines, zLines)

        # end for

        s3dGndTrBoxLines = PlotlyGraphObjects.Scatter3d(x=xLines, y=yLines, z=zLines, mode='lines')

        # make and show a plotly Figure object
        plotlyFig = PlotlyGraphObjects.Figure(data=[s3dPoints, s3dGndTrBoxLines])
        plotlyFig.update_layout(scene_aspectmode='data')
        plotlyFig.show()

        # pause here until the user presses enter
        input()
    # end for

# end function

def moveBoxFromWorldSpaceToSensorSpace(nusc: NuScenes, box: Box, lidarTopData: dict) -> Box:

    box = box.copy()

    # world space to car space
    egoPoseData: dict = nusc.get('ego_pose', lidarTopData['ego_pose_token'])
    box.translate(-np.array(egoPoseData['translation']))
    box.rotate(pyquaternion.Quaternion(egoPoseData['rotation']).inverse)

    # car space to sensor space
    calSensorData: dict = nusc.get('calibrated_sensor', lidarTopData['calibrated_sensor_token'])
    box.translate(-np.array(calSensorData['translation']))
    box.rotate(pyquaternion.Quaternion(calSensorData['rotation']).inverse)

    return box
# end function

def addLineToPlotlyLines(point1, point2, xLines: List, yLines: List, zLines: List) -> None:
    xLines.append(point1[0])
    xLines.append(point2[0])
    xLines.append(None)

    yLines.append(point1[1])
    yLines.append(point2[1])
    yLines.append(None)

    zLines.append(point1[2])
    zLines.append(point2[2])
    zLines.append(None)
# end function

if __name__ == '__main__':
    main()


