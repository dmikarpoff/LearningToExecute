cmake_minimum_required(VERSION 2.8)

project(LearningToExecute)

set(CMAKE_BUILD_TYPE Debug)

add_definitions(-std=c++11)

find_path (PUGIXML_INCLUDE_DIR
           NAMES pugixml.hpp
           PATHS ${PUGIXML_HOME}/include)
find_library (PUGIXML_LIBRARY
              NAMES pugixml
              PATHS ${PUGIXML_HOME}/lib)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (NOT (${PUGIXML_INCLUDE_DIR} STREQUAL ""))
    message(STATUS "PugiXML include = ${PUGIXML_INCLUDE_DIR}")
    message(STATUS "PugiXML library = ${PUGIXML_LIBRARY}")
else ()
    message(STATUS "No PugiXML found")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCV REQUIRED core)


set(src			src/main.cpp
                        src/ConfigurationHandler.cpp
                        src/Network.cpp
                        src/Generator.cpp
                        src/VariableManager.cpp
                        src/SymbolManager.cpp)
set(headers             include/ConfigurationHandler.h
                        include/Network.h
                        include/Generator.h
                        include/ILearningStrategy.h
                        include/BaselineStrategy.h
                        include/VariableManager.h
                        include/SymbolManager.h)

add_executable(bin/learn_to_exec 	${src}
                                        ${headers})

target_link_libraries(bin/learn_to_exec         ${PUGIXML_LIBRARY}
                                                ${OpenCV_LIBS})
