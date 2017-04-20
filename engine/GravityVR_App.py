#!/bin/env python

# file qt_pyside_app.py

import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import Qt, QTimer
from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtOpenGL import QGLWidget, QGLFormat

"""
Toy PySide application for use with "GravityVR" examples demonstrating pyopenvr
"""

class MyGlWidget(QGLWidget):
    "PySideApp uses Qt library to create an opengl context, listen to keyboard events, and clean up"

    def __init__(self, renderer, glformat, app, scene):
        self.scene = scene
        "Creates an OpenGL context and a window, and acquires OpenGL resources"
        super(MyGlWidget, self).__init__(glformat)
        self.renderer = renderer
        self.app = app
        # Use a timer to rerender as fast as possible
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.render_vr)
        # Accept keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

    def __enter__(self):
        "setup for RAII using 'with' keyword"
        return self

    def __exit__(self, type_arg, value, traceback):
        "cleanup for RAII using 'with' keyword"
        self.dispose_gl()

    def initializeGL(self):
        if self.renderer is not None:
            self.renderer.init_gl()
        self.timer.start()

    def paintGL(self):
        "render scene one time"
        self.renderer.render_scene()
        self.swapBuffers() # Seems OK even in single-buffer mode

    def render_vr(self):
        self.makeCurrent()
        self.paintGL()
        self.doneCurrent()
        self.timer.start() # render again real soon now

    def disposeGL(self):
        if self.renderer is not None:
            self.makeCurrent()
            self.renderer.dispose_gl()
            self.doneCurrent()

    def keyPressEvent(self, event):
        key = event.key()
        step = 0.125 / self.scene.mesh.size_scale

        if key == Qt.Key_Escape:
            self.app.quit()
        elif key == Qt.Key_W: #forward "w"
            self.scene.mesh.y_offset += step
        elif key == Qt.Key_S: #back "s"
            self.scene.mesh.y_offset -= step
        elif key == Qt.Key_D: #right "d"
            self.scene.mesh.x_offset += step
        elif key == Qt.Key_A: #left "a"
            self.scene.mesh.x_offset -= step
        elif key == Qt.Key_R: #up "r"
            self.scene.mesh.z_offset += step
        elif key == Qt.Key_F: #down "f"
            self.scene.mesh.z_offset -= step
        elif key == Qt.Key_Equal: #scale up "+"
            self.scene.mesh.size_scale += self.scene.mesh.size_scale/10
            self.scene.mesh.initialize = True
        elif key == Qt.Key_Minus: #scale down "-"
            self.scene.mesh.size_scale -= self.scene.mesh.size_scale/10
            self.scene.mesh.initialize = True
        elif key == Qt.Key_Up: #speed down "up arrow"
            self.scene.mesh.gravity.time_scale += 10
        elif key == Qt.Key_Down: #speed down "down arrow"
            if self.scene.mesh.gravity.time_scale >= 10:
                self.scene.mesh.gravity.time_scale -= 10 - 0.1
        elif key == Qt.Key_Space: #reset universe
            self.scene.mesh.gravity.__reset_universe__()
            self.scene.mesh.initialize = True #any hidden verts will now be re-shown

class QtPysideApp(QApplication):
    def __init__(self, renderer, scene, title):
        QApplication.__init__(self, sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle(title)
        self.window.resize(800,600)

        # Get OpenGL 4.1 context
        glformat = QGLFormat()
        glformat.setVersion(4, 1)
        glformat.setProfile(QGLFormat.CoreProfile)
        glformat.setDoubleBuffer(False)

        self.glwidget = MyGlWidget(renderer, glformat, self, scene)
        self.window.setCentralWidget(self.glwidget)

        self.window.show()

    def __enter__(self):
        "setup for RAII using 'with' keyword"
        return self

    def __exit__(self, type_arg, value, traceback):
        "cleanup for RAII using 'with' keyword"
        self.glwidget.disposeGL()

    def run_loop(self):
        retval = self.exec_()
        sys.exit(retval)

if __name__ == "__main__":
    from openvr.gl_renderer import OpenVrGlRenderer
    from openvr.color_cube_actor import ColorCubeActor
    actor = ColorCubeActor()
    renderer = OpenVrGlRenderer(actor)
    with QtPysideApp(renderer, "PySide OpenVR color cube") as qtPysideApp:
        qtPysideApp.run_loop()
