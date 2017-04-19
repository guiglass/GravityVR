import sys
import time
import numpy as np
from datetime import timedelta

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

from builder.prebuilds import get_scene_list
from engine.gravity_vectorized import newtonianLawOfGravitation

class ScatterWidget(QtGui.QWidget):
    datelabel = None
    runningtime = None

    opts = { #options used for resetting the viewport
        'center': QtGui.QVector3D(0, 0, 0),  ## will always appear at the center of the widget
        'distance': 100.0,  ## distance of camera from center
        'fov': 60,  ## horizontal field of view in degrees
        'elevation': 30,  ## camera's angle of elevation in degrees
        'azimuth': 45,  ## camera's azimuthal angle in degrees
        ## (rotation around z-axis 0 points along x-axis)
    }

    def __init__(self, builder):
        super(ScatterWidget, self).__init__()
        self.gravity = newtonianLawOfGravitation(builder)
        #Build the Qt GUI
        self.array_size = 0 #The currently loaded points (keeps qt/gl from crashing by keeping array size unchanged when verts get removed )
        self.vBox = QtGui.QHBoxLayout(self)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setMinimumSize(250, 250)
        self.init_viewport()
        self.gl_widget.show()
        self.gl_grid = gl.GLGridItem()
        self.gl_grid.setSpacing(100,100)
        self.gl_grid.setSize(10000,10000)

        self.gl_widget.addItem(self.gl_grid)
        self.vBox.addWidget(self.gl_widget)

        self.initPlots()

        t = QtCore.QTimer(self)
        t.timeout.connect(self.update)
        self.gravity.__reset_timers__()
        t.start()

    def initPlots(self):
        # initialize the scatter plot with some points
        if self.gravity.parts_coord is not None:#if there are also any particles to render
            pos = np.append(  self.gravity.verts_coord,  self.gravity.parts_coord, axis=0)
            size = np.append( self.gravity.verts_radius, self.gravity.parts_radius, axis=0)
            color = np.append(self.gravity.verts_color,  self.gravity.parts_color, axis=0)
        else: #only render vertex bodies
            pos =  self.gravity.verts_coord
            size = self.gravity.verts_radius
            color = self.gravity.verts_color
        pt_size = (size * 2) / self.gravity.size_scale
        self.array_size = pos.shape[0]
        self.sp2 = gl.GLScatterPlotItem(pos=pos, size=pt_size, color=color, pxMode=False) #pxMode false so the points in the viewport remain an absolute size
        self.sp2.setGLOptions('translucent')
        self.gl_widget.addItem(self.sp2)

    def update(self):
        out, col = self.gravity.update()

        if self.array_size - out.shape[0] > 0:
            out = np.pad(out, ((0,self.array_size - out.shape[0]),(0,0)), mode='constant')
            col = np.pad(col, ((0,self.array_size - col.shape[0]),(0,0)), mode='constant')

        self.sp2.setData(pos=out, color=col)
        epoch = self.gravity.simStartTime + self.gravity.simTotalTime

        self.runningtime.setText(str(timedelta(seconds=int(self.gravity.simTotalTime))))
        self.datelabel.setText(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch)))

    def set_time_scale(self, value):
        self.gravity.time_scale = value

    def init_viewport(self):
        for k in self.opts.keys():
            self.gl_widget.opts[k] = self.opts[k]

    def reset_universe(self):
        self.init_viewport()
        self.gravity.__reset_universe__()

class MainApp(QtGui.QWidget):
    def __init__(self, builder):
        super(MainApp, self).__init__()
        self.scatter_widget = ScatterWidget(builder)
        self.initUI()

    def initUI(self):


        vbox = QtGui.QVBoxLayout()
        hbox_1 = QtGui.QHBoxLayout()

        sld_timescale_label = QtGui.QLabel("Timescale")
        self.sld_timescale = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sld_timescale.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sld_timescale.setRange(1, 10000)
        self.sld_timescale.setValue(1)
        self.sld_timescale.valueChanged[int].connect(self.scatter_widget.set_time_scale)
        hbox_1.addWidget(sld_timescale_label)
        hbox_1.addWidget(self.sld_timescale)

        btn_reset = QtGui.QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_universe)

        lbl_datetime = QtGui.QLabel("Simulation Date")
        lbl_datetime.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl_datetime.setFixedHeight(20)
        self.lbl_datetime_txt = QtGui.QLabel("")
        self.lbl_datetime_txt.setFixedHeight(20)
        self.scatter_widget.datelabel = self.lbl_datetime_txt
        hbox_datetime = QtGui.QHBoxLayout()
        hbox_datetime.addWidget(lbl_datetime)
        hbox_datetime.addWidget(self.lbl_datetime_txt)

        lbl_runningtime = QtGui.QLabel("Simulation Time")
        lbl_runningtime.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl_runningtime.setFixedHeight(20)
        self.lbl_runningtime_txt = QtGui.QLabel("")
        self.lbl_runningtime_txt.setFixedHeight(20)
        self.scatter_widget.runningtime = self.lbl_runningtime_txt
        hbox_runningtime = QtGui.QHBoxLayout()
        hbox_runningtime.addWidget(lbl_runningtime)
        hbox_runningtime.addWidget(self.lbl_runningtime_txt)

        vbox.addLayout(hbox_datetime)
        vbox.addLayout(hbox_runningtime)
        vbox.addLayout(hbox_1)
        vbox.addWidget(btn_reset)

        vbox.addWidget(self.scatter_widget)

        self.setLayout(vbox)
        self.setWindowTitle('Law Of Gravitation Demo')
        self.show()

    def reset_universe(self):
        self.sld_timescale.setValue(1)
        self.scatter_widget.reset_universe()




if __name__ == '__main__':
    txt = "Please choose a scene number:\n"
    for v in get_scene_list():
        txt += v[0]+"\n"
    builder = get_scene_list()[int(input(txt))-1][1]

    app = QtGui.QApplication([])

    ex = MainApp(builder)
    ex.show()
    sys.exit(app.exec_())
