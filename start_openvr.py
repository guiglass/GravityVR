#!/bin/env python

from engine.GravityVR_App import QtPysideApp
from builder.prebuilds import get_scene_list
from engine.gl_renderer import OpenVrGlRenderer
from engine.scene_actor import SceneActor

"""
PySide application for use with "GravityVR" examples demonstrating pyopenvr
"""

if __name__ == "__main__":
    txt = "Please choose a scene number:\n"
    for v in get_scene_list():
        txt += v[0]+"\n"
    builder = get_scene_list()[int(input(txt))-1][1]

    scene = SceneActor(builder)

    renderer = OpenVrGlRenderer()
    renderer.append(scene)

    from engine.tracked_devices_actor import TrackedDevicesActor
    renderer.append(TrackedDevicesActor(renderer.poses))

    with QtPysideApp(renderer, scene, "OpenVR Gravitation Demo") as qtPysideApp:
        qtPysideApp.run_loop()