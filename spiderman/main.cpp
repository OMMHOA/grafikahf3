#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>        // must be downloaded 
#include <GL/freeglut.h>    // must be downloaded unless you have an Apple
#endif
 
const unsigned int windowWidth = 600, windowHeight = 600;
 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...
 
// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;
 
void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete[] log;
    }
}
 
// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}
 
// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}
 
struct mat4 {
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }
 
    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
 
    void SetUniform(unsigned shaderProg, char * name) {
        int loc = glGetUniformLocation(shaderProg, name);
        glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
    }
 
    operator float*() { return &m[0][0]; }
};
 
mat4 Translate(float tx, float ty, float tz) {
    return mat4(1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        tx, ty, tz, 1);
}
 
mat4 Rotate(float angle, float wx, float wy, float wz) {
    float c = cosf(angle);
    float s = sinf(angle);
    return mat4(c*(1 - wx*wx) + wx*wx, wx*wy*(1 - c) + s*wz, wx*wz*(1 - c) - s*wy, 0,
        wy*wx*(1 - c) - s*wz, c*(1 - wy*wy) + wy*wy, wx*wz*(1 - c) + s*wx, 0,
        wz*wx*(1 - c) + s*wy, wz*wy*(1 - c) - s*wx, c*(1 - wz*wz) + wz*wz, 0,
        0, 0, 0, 1);
}
 
mat4 Scale(float sx, float sy, float sz) {
    return mat4(
        sx, 0, 0, 0,
        0, sy, 0, 0,
        0, 0, sz, 0,
        0, 0, 0, 1);
}
 
// 3D point in homogeneous coordinates
struct vec4 {
    float v[4];
 
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
 
    void SetUniform(unsigned int shaderProg, char * name) {
        int loc = glGetUniformLocation(shaderProg, name);
        glUniform4f(loc, v[0], v[1], v[2], v[3]);
    }
    vec4 operator+(const vec4& ve){
        return vec4(this->v[0]+ve.v[0], this->v[1]+ve.v[1], this->v[2]+ve.v[2], this->v[3]+ve.v[3]);
    }
};
 
struct vec3 {
    float x, y, z;
 
    vec3(float x0 = 0, float y0 = 0, float z0 = 0) {
        x = x0; y = y0; z = z0;
    }
    vec3(vec4 v) {
        x = v.v[0];
        y = v.v[1];
        z = v.v[2];
    }
    vec3 operator*(float a) {
        return vec3(x * a, y * a, z * a);
    }
    vec3 operator+(const vec3& v) {
        return vec3(x + v.x, y + v.y, z + v.z);
    }
    vec3 operator-(const vec3& v) {
        return vec3(x - v.x, y - v.y, z - v.z);
    }
    float operator*(const vec3& v) {     // dot product
        return (x * v.x + y * v.y + z * v.z);
    }
    vec3 operator%(const vec3& v) {     // cross product
        return vec3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
    }
    vec3 operator/(const float f){
        return vec3(x/f,y/f,z/f);
    }
    float Length() { return sqrt(x * x + y * y + z * z); }
    vec3 normalize() {
        float l = this->Length();
        return vec3(x / l, y / l, z / l);
    }
    void SetUniform(unsigned int shaderProg, char* name) {
        int loc = glGetUniformLocation(shaderProg, name);
        glUniform3f(loc, x, y, z);
    }
};
 
struct LightSource {
    vec3 La, Le;
    vec4 wLiPos;
    LightSource(){}
    LightSource(vec4 pos, vec3 la, vec3 le) :wLiPos(pos), La(la), Le(le){}
    void Animate(float t){
    }
};
 
struct RenderState {
    mat4 M, Minv, MVP;
    vec3 kd, ks, ka, wEye, wLookat, wVup;
    LightSource lSource1, lSource2;
    float shine;
    float fov, asp, fp, bp;
 
    RenderState() {
        lSource1 = LightSource(vec4(10, 0, 0), vec3(0.03f, 0.02f, 0.01f), vec3(0.3f, 0.3f, 0));
        lSource2 = LightSource(vec4(-10, 0, 10), vec3(0.01f, 0.02f, 0.03f), vec3(0, 0.5f, 0.5f));
        kd = vec3(1, 1, 1);
        ks = vec3(1, 1, 1);
        ka = vec3(0, 0, 0);
        shine = 0.1f;
        fov = 1;
        fp = 1;
        asp = 1;
        bp = 100;
        wEye = vec3(8, 0, 8);
        wLookat = vec3(0, 0, 0);
        wVup = vec3(0, 1, 0);
    }
};
 
RenderState state;
 
struct Shader {
    unsigned int shaderProg;
 
    void Create(const char * vsSrc, const char * vsAttrNames[],
        const char * fsSrc, const char * fsOuputName) {
        unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
        if (!vs) {
            printf("Error in vertex shader creation\n");
            exit(1);
        }
        glShaderSource(vs, 1, &vsSrc, NULL);
        glCompileShader(vs);
        checkShader(vs, "Vertex shader error 1");
        unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fs) {
            printf("Error in fragment shader creation\n");
            exit(1);
        }
        glShaderSource(fs, 1, &fsSrc, NULL);
        glCompileShader(fs);
        checkShader(fs, "Fragment shader error");
 
        shaderProg = glCreateProgram();
        if (!shaderProg) {
            printf("Error in shader program creation\n");
            exit(1);
        }
        glAttachShader(shaderProg, vs);
        glAttachShader(shaderProg, fs);
 
        for (int i = 0; i < 2; i++)
            glBindAttribLocation(shaderProg, i, vsAttrNames[i]);
        glBindFragDataLocation(shaderProg, 0, fsOuputName);
        glLinkProgram(shaderProg);
        checkLinking(shaderProg);
    }
};
 
struct Phong : public Shader {
    const char * vsSrc = R"(
        #version 140
        precision highp float;
        uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
        uniform vec4  wLiPos1, wLiPos2;       // pos of light source 
        uniform vec3  wEye;         // pos of eye
 
                                                in  vec3 vtxPos;            // pos in modeling space
        in  vec3 vtxNorm;           // normal in modeling space
        in  vec2 vtxUV;
 
                                                out vec3 wNormal;           // normal in world space
        out vec3 wView;             // view in world space
        out vec3 wLight1;            // light dir in world space
        out vec3 wLight2;
        out vec2 texcoord;
 
                                                void main() {
           gl_Position = vec4(vtxPos.xyz, 1) * MVP; // to NDC
           texcoord = vtxUV;
           vec4 wPos = vec4(vtxPos.xyz, 1) * M;
           wLight1  = wLiPos1.xyz * wPos.w - wPos.xyz * wLiPos1.w;
           wLight2 = wLiPos2.xyz * wPos.w - wPos.xyz * wLiPos2.w;
           wView   = wPos.xyz - wEye * wPos.w;
           wNormal = (Minv * vec4(vtxNorm.xyz, 0)).xyz;
        }
    )";
 
    const char * fsSrc = R"(
        #version 140
        precision highp float;
        uniform sampler2D samplerUnit;
        uniform vec3 kd, ks, ka;// diffuse, specular, ambient ref
        uniform vec3 La1, La2, Le1, Le2;    // ambient and point source rad
        uniform float shine;    // shininess for specular ref
 
                                                                                                                                                                                                        in vec2 texcoord;
        in  vec3 wNormal;       // interpolated world sp normal
        in  vec3 wView;         // interpolated world sp view
        in  vec3 wLight1;        // interpolated world sp illum dir
        in  vec3 wLight2;
        out vec4 fragmentColor; // output goes to frame buffer
 
                                                                                                                                                                                                void main() {
           vec3 N = normalize(wNormal);
           vec3 V = normalize(wView); 
           vec3 L1 = normalize(wLight1);
           vec3 L2 = normalize(wLight2);
           vec3 H1 = normalize(L1 + V);
           vec3 H2 = normalize(L2 + V);
           float cost1 = max(dot(N,L1), 0), cosd1 = max(dot(N,H1), 0);
           float cost2 = max(dot(N,L2), 0), cosd2 = max(dot(N,H2), 0);
           vec3 color = ka * La1 +  (kd * cost1 + ks * pow(cosd1,shine)) * Le1;
           color += ka * La2 +  (kd * cost2 + ks * pow(cosd2,shine)) * Le2;
           fragmentColor = vec4(color,1) + texture(samplerUnit, texcoord);
        }
    )";
public:
    void Create() {
        static const char * vsAttrNames[] = { "vtxPos", "vtxNorm" };
        Shader::Create(vsSrc, vsAttrNames, fsSrc, "fragmentColor");
    }
};
 
// handle of the shader program
Phong phong;
 
void BindUniforms() {
    glUseProgram(phong.shaderProg);
    state.MVP.SetUniform(phong.shaderProg, "MVP");
    state.M.SetUniform(phong.shaderProg, "M");
    state.Minv.SetUniform(phong.shaderProg, "Minv");
    state.kd.SetUniform(phong.shaderProg, "kd");
    state.ks.SetUniform(phong.shaderProg, "ks");
    state.ka.SetUniform(phong.shaderProg, "ka");
    state.lSource1.La.SetUniform(phong.shaderProg, "La1");
    state.lSource1.Le.SetUniform(phong.shaderProg, "Le1");
    state.lSource1.wLiPos.SetUniform(phong.shaderProg, "wLiPos1");
    state.lSource2.La.SetUniform(phong.shaderProg, "La2");
    state.lSource2.Le.SetUniform(phong.shaderProg, "Le2");
    state.lSource2.wLiPos.SetUniform(phong.shaderProg, "wLiPos2");
    state.wEye.SetUniform(phong.shaderProg, "wEye");
    glUniform1f(glGetUniformLocation(phong.shaderProg, "shine"), state.shine);
}
 
struct Texture {
    unsigned int textureId;
    float *image;
 
    Texture() {
        image = new float[100 * 100 * 3];
    }
    void Generate() {
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        float *img = image;
        int width=100, height=100;
        for (int i = 0; i<width; i++) {
            for (int j = 0; j<height; j++) {
                if ((i / 5) % 2 == 0) {
                    if ((j / 5) % 2 == 0) {
                        *img++ = 1;
                        *img++ = 1;
                        *img++ = 1;
                    }
                    else {
                        *img++ = 0;
                        *img++ = 0;
                        *img++ = 0;
                    }
                }
                else {
                    if ((j / 5) % 2 == 0) {
                        *img++ = 0;
                        *img++ = 0;
                        *img++ = 0;
                    }
                    else {
                        *img++ = 1;
                        *img++ = 1;
                        *img++ = 1;
                    }
                }
            }
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
            0, GL_RGB, GL_FLOAT, image); //Texture -> OpenGL
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    ~Texture(){
        delete image;
    }
};
 
Texture texture;
 
struct Camera {
    Camera(){
    }
    mat4 V() { // view matrix
        vec3 w = (state.wEye - state.wLookat).normalize();
        vec3 u = (state.wVup%w).normalize();
        vec3 v = (w%u);
        return Translate(-state.wEye.x, -state.wEye.y, -state.wEye.z) *
            mat4(u.x, v.x, w.x, 0.0f,
                u.y, v.y, w.y, 0.0f,
                u.z, v.z, w.z, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f);
    }
    mat4 P() { // projection matrix
        float sy = 1 / tan(state.fov / 2);
        return mat4(sy / state.asp, 0.0f, 0.0f, 0.0f,
            0.0f, sy, 0.0f, 0.0f,
            0.0f, 0.0f, -(state.fp + state.bp) / (state.bp - state.fp), -1.0f,
            0.0f, 0.0f, -2 * state.fp*state.bp / (state.bp - state.fp), 0.0f);
    }
};
 
// 2D camera
Camera camera;
 
struct Geometry {
    unsigned int vao, nVtx;
    vec3 scale, pos, rotAxis;
    float rotAngle;
 
    Geometry() : scale(vec3(1, 1, 1)), pos(vec3(0, 0, 0)), rotAxis(vec3(0, 0, 1)), rotAngle(0) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
    void Draw() {
        mat4 M = Scale(scale.x, scale.y, scale.z) *
            Rotate(rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
            Translate(pos.x, pos.y, pos.z);
        mat4 Minv = Translate(-pos.x, -pos.y, -pos.z) *
            Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
            Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
        mat4 MVP = M * camera.V() * camera.P();
        state.M = M;
        state.Minv = Minv;
        state.MVP = MVP;
        BindUniforms();
 
        int samplerUnit = 0;
        glUniform1i(glGetUniformLocation(phong.shaderProg, "samplerUnit"), samplerUnit);
        glActiveTexture(GL_TEXTURE0 + samplerUnit);
        glBindTexture(GL_TEXTURE_2D, texture.textureId);
 
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, nVtx);
    }
};
 
struct VertexData {
    vec3 position, normal;
    float u, v;
};
 
struct ParamSurface : public Geometry {
    virtual VertexData GenVertexData(float u, float v) = 0;
    void Create(int N, int M);
};
 
void ParamSurface::Create(int N, int M) {
    nVtx = N * M * 6;
    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
 
    VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;
    for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) {
        *pVtx++ = GenVertexData((float)i / N, (float)j / M);
        *pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
        *pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
 
        *pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
        *pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
        *pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
    }
 
    int stride = sizeof(VertexData), sVec3 = sizeof(vec3);
    glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);
 
    glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
    glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
    glEnableVertexAttribArray(2);  // AttribArray 2 = UV
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)sVec3);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sVec3));
    delete[] vtxData;
}
 
class Torus : public ParamSurface {
    vec3 center;
    float r, R;
public:
    Torus(){}
    Torus(vec3 c, float radius, float Radius) : center(c), r(radius), R(Radius) {
        ParamSurface::Create(128,64);
    }
    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        float cu = cosf(2 * M_PI*u), cv = cosf(2 * M_PI*v), su = sinf(2*M_PI*u), sv = sinf(2*M_PI*v);
        vd.position = vec3((R + r*cu)*cv, r*su, (R+r*cu)*sv);
        vd.normal = (vec3(R* cv,0,R * sv) - vd.position).normalize();
        vd.u = u;
        vd.v = v;
        return vd;
    }
    void SetValues(vec3 c, float radius, float Radius){ 
        center = c; 
        r = radius;
        R = Radius;
    }
};
 
VertexData TorusGenVertexData(float u, float v, vec3 pos, float r, float R) {
    VertexData vd;
    float cu = cosf(2 * M_PI*u), cv = cosf(2 * M_PI*v), su = sinf(2 * M_PI*u), sv = sinf(2 * M_PI*v);
    vd.position = vec3((R + r*cu)*cv, r*su, (R + r*cu)*sv);
    vd.normal = (vec3(R* cv, 0, R * sv) - vd.position).normalize();
    vd.u = u;
    vd.v = v;
    return vd;
}
float xangle = 0, yangle = 0;
struct Sphere : public ParamSurface {
    vec3 center;
    float radius;
    bool shouldFollow = true;
    Sphere(vec3 c, float r) : center(c), radius(r){
        ParamSurface::Create(32, 16);
    }
    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.normal = vec3(cos(u * 2 * M_PI) * sin(v*M_PI),
            sin(u * 2 * M_PI) * sin(v*M_PI),
            cos(v*M_PI));
        vd.position = vd.normal * radius + center;
        vd.u = u; vd.v = v;
        return vd;
    }
    void Animate(float t) {
        float u = sinf(t/2800)/5;
        float v = t/3000;
        VertexData vd = TorusGenVertexData(u, v, vec3(0,0,0), 9.9, 10);
 
        //rotAxis = (vel%vd.normal).normalize();
        //rotAngle += 0.01f;
        center = vd.position + vd.normal * radius;
        if(shouldFollow){
            state.wLookat = center;
        
            vec3 eye_sphere_vector = (center - state.wEye).normalize();
            eye_sphere_vector.y = 0;
            vec3 eye_origo_vector = (center * (-1.0f)).normalize();
            eye_origo_vector.y = 0;
            // TODO calculate angle correctly!
            xangle = asinf(eye_sphere_vector * eye_origo_vector);
            printf("%f \n",xangle);
        }
        ParamSurface::Create(32,16);
    }
};
 
struct Scene {
    Sphere sphere;
    Torus torus;
 
    Scene() : sphere(Sphere(vec3(0, 0, 0), 1)), torus(Torus(vec3(0, 0, 0), 9.9, 10)) {
    }
 
    void Draw() {
        torus.Draw();
        sphere.Draw();
    }
 
    void Animate(float t) {
        sphere.Animate(t);
        state.lSource1.Animate(t);
        state.lSource2.Animate(t);
    }
};
 
Scene * scene;
 
// Initialization, create an OpenGL context
void onInitialization() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glViewport(0, 0, windowWidth, windowHeight);
    phong.Create();
    texture.Generate();
    scene = new Scene();
}
 
void onExit() {
    glDeleteProgram(phong.shaderProg);
    delete scene;
    printf("exit");
}
 
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.93f, 0.93f, 0.93f, 0);                            // background color 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene->Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'a') {
        xangle -= M_PI / 16;
        printf("%f\n",xangle);
        state.wLookat = vec3(8*cosf(xangle)+8, 0, 8*sinf(xangle)+8);
        glutPostRedisplay();         // if d, invalidate display, i.e. redraw
    } else if (key == 'd'){
        xangle += M_PI / 16;
        state.wLookat = vec3(8*cosf(xangle)+8, 0, 8*sinf(xangle)+8);
        glutPostRedisplay();         // if d, invalidate display, i.e. redraw
    }
    if (key == ' ') {
        if(scene->sphere.shouldFollow)
            scene->sphere.shouldFollow = false;
        else
            scene->sphere.shouldFollow = true;
    }
    if (key =='w') {
        yangle -= M_PI / 16;
        state.wLookat = vec3(8, 8*sinf(yangle)+8, 8*cosf(yangle)+8);
        glutPostRedisplay();
    } else if (key == 's') {
        yangle += M_PI / 16;
        state.wLookat = vec3(8, 8*sinf(yangle)+8, 8*cosf(yangle)+8);
        glutPostRedisplay();
    }
}
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
 
}
 
// Mouse click event
void onMouse(int button, int _state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && _state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        glutPostRedisplay();     // redraw
    }
}
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
 
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    scene->Animate(time);
    glutPostRedisplay();                    // redraw the scene
}
 
// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);                // Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);                            // Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);
 
#if !defined(__APPLE__)
    glewExperimental = true;    // magic
    glewInit();
#endif
 
    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
 
    onInitialization();
 
    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);
 
    glutMainLoop();
    onExit();
    return 1;
}