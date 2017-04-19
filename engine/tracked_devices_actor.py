#!/bin/env python

# file tracked_devices_actor.py

import time
from textwrap import dedent
from ctypes import cast, c_float, c_void_p, sizeof

import numpy as np
from OpenGL.GL import *  # @UnusedWildImport # this comment squelches an IDE warning
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.arrays import vbo
from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT, GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT

import openvr
from openvr.gl_renderer import matrixForOpenVrMatrix

"""
Tracked item (controllers, lighthouses, etc) actor for "hello world" openvr apps
"""

class TrackedDeviceMesh(object):

    pulse = 0
    array_size = 350

    def __init__(self, model_name):
        "This constructor must only be called with a live OpenGL context"
        #initalize numpy arrays
        self.vertices = self.get_vertices() #get vertex coords from a custom function
        self.colors = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), (self.array_size,1)).astype(np.float32) #Each row is for a vertex in RGB
        self.colors[::10] -= 0.8 #FUN STUFF makes every nth vert real big

        self.sizes = np.ones(self.array_size, dtype=np.float32) * 50 #the point size for each vertex
        self.sizes[::10]  += 20 #FUN STUFF makes every nth vert real big
        self.indices = np.arange(self.array_size, dtype=np.uint32) #an index for each vertex in the vertices array (we want to update all verts each frame)

        self.vertexPositions = vbo.VBO(self.vertices) #Create a VBO for each vert's positions in 3d
        self.vertexColors = vbo.VBO(self.colors) #Create a VBO for each vert's point color
        self.vertexSizes = vbo.VBO(self.sizes) #Create a VBO for each vert's point size
        self.indexPositions = vbo.VBO(self.indices, target=GL_ELEMENT_ARRAY_BUFFER) #The necessary VOB for the indices
        # http://stackoverflow.com/questions/14365484/how-to-draw-with-vertex-array-objects-and-gldrawelements-in-pyopengl
        self.vao = glGenVertexArrays(1) #create the VAO
        glBindVertexArray(self.vao) #start bind with VAO

        # Colors data buffer initialization (only touched once)
        self.vertexColors = vbo.VBO(self.colors)
        self.vertexColors.bind()
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, False, 0, None)

        # Sizes data buffer initialization (only touched once)
        self.vertexSizes = vbo.VBO(self.sizes)
        self.vertexSizes.bind()
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, False, 0, None)

        self.indexPositions.bind() #bind indices buffer
        glBindVertexArray(0) #stop bind VAO

    def get_vertices(self):
        ##############################################################################
        # vertices
        ##############################################################################

        theta = np.linspace(-4 * np.pi, 4 * np.pi, self.array_size)
        z = np.linspace(-2, 2, self.array_size)
        r = z**2 + 1
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return np.dstack((x,y,z)).astype(np.float32) * 0.1


    def display_gl(self, modelview, projection, pose):

        self.pulse += 0.0125 #make the model pulse in size

        controller_X_room = pose.mDeviceToAbsoluteTracking
        controller_X_room = matrixForOpenVrMatrix(controller_X_room)
        modelview0 = controller_X_room * modelview
        # Repack before use, just in case
        modelview0 = np.asarray(np.matrix(modelview0, dtype=np.float32))
        glUniformMatrix4fv(4, 1, False, modelview0)

        glBindVertexArray(self.vao) #start bind with VAO

        # Vertices data buffer initialization (only touched once)

        verts_pulse = self.vertices * ( 1 + np.sin(self.pulse) * 0.35 )
        self.vertexPositions = vbo.VBO(verts_pulse.astype(np.float32))
        self.vertexPositions.bind()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)

        glDrawElements(GL_POINTS, self.indexPositions.size, GL_UNSIGNED_INT, None)
        glBindVertexArray(0) #stop bind VAO

    def dispose_gl(self):
        glDeleteVertexArrays(1, (self.vao,))
        self.vbo = 0
        self.vertexPositions.delete()
        self.indexPositions.delete()


class TrackedDevicesActor(object):
    """
    Draws Vive controllers and lighthouses.
    """

    def __init__(self, pose_array):
        self.shader = 0
        self.poses = pose_array
        self.meshes = dict()
        self.show_controllers_only = True

    def _check_devices(self):
        "Enumerate OpenVR tracked devices and check whether any need to be initialized"
        for i in range(1, len(self.poses)):
            pose = self.poses[i]
            if not pose.bDeviceIsConnected:
                continue
            if not pose.bPoseIsValid:
                continue
            if self.show_controllers_only:
                device_class = openvr.VRSystem().getTrackedDeviceClass(i)
                if not device_class == openvr.TrackedDeviceClass_Controller:
                    continue
            model_name = openvr.VRSystem().getStringTrackedDeviceProperty(i, openvr.Prop_RenderModelName_String)
            # Create a new mesh object, if necessary
            if not model_name in self.meshes:
                self.meshes[model_name] = TrackedDeviceMesh(model_name)

    def init_gl(self):
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) #allow the program to specify the point size

        vertex_shader = compileShader(dedent(
            """\
            #version 450 core
            #line 40

            layout(location = 0) in vec3 in_Position;
            layout(location = 1) in vec4 in_Color;
            layout(location = 2) in float in_Size;

            layout(location = 0) uniform mat4 projection = mat4(1);
            layout(location = 4) uniform mat4 model_view = mat4(1);

            out vec4 _color;

            void main() {
                gl_Position = projection * model_view * vec4(in_Position, 1.0);
                _color = in_Color; // color by texture coordinate

                vec3 ndc = gl_Position.xyz / gl_Position.w ; // perspective divide.
                float zDist = 1.0-ndc.z ; // 1 is close (right up in your face,)
                // 0 is far (at the far plane)
                gl_PointSize = in_Size*zDist ; // between 0 and 50 now.
            }
            """),
            GL_VERTEX_SHADER)

        fragment_shader = compileShader(dedent("""
            #version 450 core
            #line 59

            in vec4 _color;
            out vec4 FragColor;

            float Ns = 250;
            vec4 mat_specular=vec4(1);
            vec4 light_specular=vec4(1);

            void main() {
                //FragColor = vec4(_color, 1.0); //old way to just pass a color to the vertex (results in a rectangle)

                //Calculate normal from texture coordinates

                vec3 N;
                N.xy = gl_PointCoord* 2.0 - vec2(1.0);
                float mag = dot(N.xy, N.xy);
                if (mag > 1.0) discard;   // kill pixels outside circle
                N.z = sqrt(1.0-mag);

                // calculate lighting

                float diffuse = max(0.0, dot(vec3(1.0,0.0,1.0), N));


                vec3 eye = vec3 (0.0, 0.0, 1.0);
                vec3 halfVector = normalize( eye + vec3(1.0,0.0,1.0));
                float spec = max( pow(dot(N,halfVector), Ns), 0.);
                vec4 S = light_specular*mat_specular* spec;

                FragColor = _color * diffuse + S;
            }
            """), GL_FRAGMENT_SHADER)
        self.shader = compileProgram(vertex_shader, fragment_shader)

    def display_gl(self, modelview, projection):
        self._check_devices()
        glEnable(GL_DEPTH_TEST)
        glUseProgram(self.shader)
        glUniformMatrix4fv(0, 1, False, projection)
        for i in range(1, len(self.poses)):
            pose = self.poses[i]
            if not pose.bPoseIsValid:
                continue
            model_name = openvr.VRSystem().getStringTrackedDeviceProperty(i, openvr.Prop_RenderModelName_String)
            if not model_name in self.meshes:
                continue # Come on, we already tried to load it a moment ago. Maybe next time.
            mesh = self.meshes[model_name]
            mesh.display_gl(modelview, projection, pose)

    def dispose_gl(self):
        glDeleteProgram(self.shader)
        self.shader = 0
        for key in list(self.meshes):
            mesh = self.meshes[key]
            mesh.dispose_gl()
            del self.meshes[key]
