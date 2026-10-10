cmake_minimum_required(VERSION 3.20)

add_library(fluca_mesh SHARED
    ib/impls/levelset/levelset.c
    ib/impls/surface/surface.c
    ib/impls/surface/surfacestl.c
    ib/interface/ibbodybasic.c
    ib/interface/ibbodypkg.c
    ib/interface/ibbodyreg.c
    impls/cartesian/cartesian.c
    impls/cartesian/cartesiancgns.c
    impls/cartesian/cartesianib.c
    interface/meshbasic.c
    interface/meshib.c
    interface/meshopts.c
    interface/meshpkg.c
    interface/meshreg.c
)

target_include_directories(fluca_mesh PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_mesh PUBLIC
    fluca::sys
    fluca::viewer
)
add_library(fluca::mesh ALIAS fluca_mesh)
