cmake_minimum_required(VERSION 3.2.0)

project(CarPlate)

include_directories(.)
include_directories(./detector)

set(OpenCV_DIR "/usr/local/opencv_3.4.7/share/OpenCV")
find_package(OpenCV REQUIRED)
if (NOT ${OPENCV_FOUND})
    message(SEND_ERROR "Failed to find OpenCV")
    return()
endif()

add_subdirectory(./3rdParty/darknet)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(./3rdParty/darknet/include)

set(SOURCES
    main.cpp
    grabber/video_grabber.cpp
    handlers/handler.cpp
    detector/yolo/cpu_yolo_detector.cpp
    detector/yolo/gpu_yolo_detector.cpp
    detector/yolo/yolo_detector.cpp
    tracker/tracker.cpp
    tracker/multi_tracker.cpp
    tracker/single_tracker.cpp
    recognizer/recognizer.cpp
    filtering/kalman/kalman_filter.cpp
    filtering/kalman/kalman_rect_tracker.cpp

    tracker/deepsort/sort.cpp
    tracker/deepsort/tracker_manager.cpp
    tracker/deepsort/tracker_kalman.cpp
)

add_executable (${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} dark pthread)

