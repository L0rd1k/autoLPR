cmake_minimum_required(VERSION 3.0.0)

project(CarPlate)

include_directories(.)
include_directories(./detector)

set(OpenCV_DIR "/usr/local/opencv_3.4.7/share/OpenCV")
find_package(OpenCV REQUIRED)
if (NOT ${OPENCV_FOUND})
    message(SEND_ERROR "Failed to find OpenCV")
    return()
endif()

set(SOURCES
    main.cpp
    grabber/video_grabber.cpp
    detector/morphology_detector.cpp
    detector/cpu_yolo_detector.cpp
    detector/detector.cpp
    tracker/tracker.cpp
)

add_executable (${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} pthread)

