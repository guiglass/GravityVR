#!/bin/env python
from textwrap import dedent

import numpy as np
from OpenGL.GL import *  # @UnusedWildImport # this comment squelches an IDE warning
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.arrays import vbo

from .gravity_vectorized import newtonianLawOfGravitation

"""
Scene for simple Newton law of gravitation in openvr example
"""

class MeshActor(object):
    size_scale = 1

    x_offset = 0
    y_offset = 0
    z_offset = 0

    initialize = True #will be false after initializing VAO arrays, if set true again the the buffers will reload

    def __init__(self, scene):
        self.gravity = newtonianLawOfGravitation(scene)
        self.array_size = self.gravity.builder.get_array_size()
        self._init_arrays()

        "This constructor must only be called with a live OpenGL context"
        self.indices = np.arange(self.array_size, dtype=np.uint32) #an index for each vertex in the vertices array (we want to update all verts each frame)
        #self.vertexPositions = vbo.VBO(self.vertices) #Create a VBO for each vert's positions in 3d
        #self.vertexColors = vbo.VBO(self.colors) #Create a VBO for each vert's point color
        self.vertexSizes = vbo.VBO(self.sizes) #Create a VBO for each vert's point size
        self.indexPositions = vbo.VBO(self.indices, target=GL_ELEMENT_ARRAY_BUFFER) #The necessary VOB for the indices

    def _init_arrays(self):
        #self.vertices = self.gravity.verts_coord
        #self.colors = self.gravity.verts_color #Each vertex in RGB
        self.sizes = self.gravity.verts_radius
        if self.gravity.parts_coord is not None:
            #self.vertices = np.append(self.vertices, self.gravity.parts_coord, axis=0)
            #self.colors = np.append(self.colors, self.gravity.parts_color, axis=0)
            self.sizes = np.append(self.sizes, self.gravity.parts_radius, axis=0)

    def _init_buffers(self):
        self.vao = glGenVertexArrays(1) #create the VAO
        glBindVertexArray(self.vao) #start bind with VAO

        # Colors data buffer initialization (only touched once unless reinitialized) / This has now been moved to the main loop
        #self.vertexColors = vbo.VBO(self.colors.astype(np.float32))
        #self.vertexColors.bind()
        #glEnableVertexAttribArray(1)
        #glVertexAttribPointer(1, 4, GL_FLOAT, False, 0, None)

        # Sizes data buffer initialization (only touched once unless reinitialized)
        self.vertexSizes = vbo.VBO((self.sizes * self.size_scale * 5500 / self.gravity.size_scale ).astype(np.float32))
        self.vertexSizes.bind()
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, False, 0, None)

        self.indexPositions.bind() #bind indices buffer
        glBindVertexArray(0) #stop bind VAO
        self.initialize = False

    def display_gl(self):
        if self.initialize:
            self._init_arrays()
            self._init_buffers()

        self.vertices, self.colors = self.gravity.update()

        self.vertices[:,0] += self.y_offset
        self.vertices[:,1] += self.z_offset
        self.vertices[:,2] += self.x_offset

        glBindVertexArray(self.vao) #start bind with VAO

        # Vertices data buffer initialization
        self.vertexPositions = vbo.VBO(self.vertices.astype(np.float32) * self.size_scale)
        self.vertexPositions.bind()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)

        # Colors data buffer initialization
        self.vertexColors = vbo.VBO(self.colors.astype(np.float32))
        self.vertexColors.bind()
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, False, 0, None)#

        glDrawElements(GL_POINTS, self.vertices.shape[0], GL_UNSIGNED_INT, None)
        glBindVertexArray(0) #stop bind VAO


    def dispose_gl(self):
        glDeleteVertexArrays(1, (self.vao,))
        self.vbo = 0
        self.vertexPositions.delete()
        self.indexPositions.delete()

class SceneActor(object):
    mesh = None

    def __init__(self, builder):
        self.builder = builder
        self.shader = 0

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

                FragColor = _color * diffuse;
            }
            """), GL_FRAGMENT_SHADER)

        self.shader = compileProgram(vertex_shader, fragment_shader)
        self.mesh = MeshActor(self.builder)

    def display_gl(self, modelview, projection):

        glEnable(GL_DEPTH_TEST)
        glUseProgram(self.shader)
        glUniformMatrix4fv(0, 1, False, projection)
        glUniformMatrix4fv(4, 1, False, modelview)

        self.mesh.display_gl()

    def dispose_gl(self):
        glDeleteProgram(self.shader)
        self.shader = 0
        self.mesh.dispose_gl()
        del self.mesh
