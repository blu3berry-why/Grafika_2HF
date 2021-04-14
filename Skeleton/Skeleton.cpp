//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Gyenese Mátyás
// Neptun : VSQUVG
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};

float kys(float n, float k);

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

struct SmoothMaterial : Material {
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Face {
	std::vector<int> vertecies;
	Face(int a, int b, int c, int d, int e) {
		vertecies = std::vector<int>(5);
		vertecies[0] = a;
		vertecies[1] = b;
		vertecies[2] = c;
		vertecies[3] = d;
		vertecies[4] = e;
	}
	Face() {}
};

struct Dodecahedron {
	std::vector<vec3> vertecies;
	std::vector<Face> faces;
	std::vector<vec3> middlepoints;
	Dodecahedron() {
		vertecies = std::vector<vec3>(20);
		faces = std::vector<Face>(12);
		middlepoints = std::vector<vec3>(12);

		vertecies[0] = vec3(0, 0.618, 1.618);
		vertecies[1] = vec3(0, -0.618, 1.618);
		vertecies[2] = vec3(0, -0.618, -1.618);
		vertecies[3] = vec3(0, 0.618, -1.618);

		vertecies[4] = vec3(1.618, 0, 0.618);
		vertecies[5] = vec3(-1.618, 0, 0.618);
		vertecies[6] = vec3(-1.618, 0, -0.618);
		vertecies[7] = vec3(1.618, 0, -0.618);

		vertecies[8] = vec3(0.618, 1.618, 0);
		vertecies[9] = vec3(-0.618, 1.618, 0);
		vertecies[10] = vec3(-0.618, -1.618, 0);
		vertecies[11] = vec3(0.618, -1.618, 0);

		vertecies[12] = vec3(1.0f, 1.0f, 1.0f);
		vertecies[13] = vec3(-1.0f, 1.0f, 1.0f);
		vertecies[14] = vec3(-1.0f, -1.0f, 1.0f);
		vertecies[15] = vec3(1.0f, -1.0f, 1.0f);

		vertecies[16] = vec3(1.0f, -1.0f, -1.0f);
		vertecies[17] = vec3(1.0f, 1.0f, -1.0f);
		vertecies[18] = vec3(-1.0f, 1.0f, -1.0f);
		vertecies[19] = vec3(-1.0f, -1.0f, -1.0f);


		faces[0] = Face(0, 1, 15, 4, 12);
		faces[1] = Face(0, 12, 8, 9, 13);
		faces[2] = Face(0, 13, 5, 14, 1);
		faces[3] = Face(1, 14, 10, 11, 15);

		faces[4] = Face(2, 3, 17, 7, 16);
		faces[5] = Face(2, 16, 11, 10, 19);
		faces[6] = Face(2, 19, 6, 18, 3);
		faces[7] = Face(18, 9, 8, 17, 3);
		
		faces[8] = Face(15, 11, 16, 7, 4);
		faces[9] = Face(4, 7, 17, 8, 12);
		faces[10] = Face(13, 9, 18, 6, 5);
		faces[11] = Face(5, 6, 19, 10, 14);
	}

};

Dodecahedron D;
const float meshWidth = 0.1f;

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir);}
	Ray() {}
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

class Sphere : public Intersectable {
	vec3 center;
	float radius;
public:
	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center; radius = _radius; material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir); 
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		hit.material = material;
		return hit;
	}

};

class Triangle : public Intersectable {
protected:
	vec3 a;
	vec3 b;
	vec3 c;
	Material* material;
public:
	Triangle(vec3 _a, vec3 _b, vec3 _c, Material* _material) {
		a = _a;
		b = _b;
		c = _c;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 n = cross(b - a, c - a);
		float t = dot((a - ray.start), n) / dot(ray.dir, n);
		if (t < 0) return hit;
		vec3 p = ray.start + ray.dir * t;
		float q1 = dot(cross(b - a, p - a), n);
		float q2 = dot(cross(c - b, p - b), n);
		float q3 = dot(cross(a - c, p - c), n);
		if (q1 > 0 && q2 > 0 && q3 > 0) {
			hit.t = t;
			hit.normal = normalize(n);
			hit.position = ray.start + ray.dir * t;
			hit.material = material;
		}
		return hit;
	}
};

class DiffuseTriangle : public Triangle {
public:
	DiffuseTriangle(vec3 a, vec3 b, vec3 c, Material* _material) : Triangle(a, b, c, _material){
		
	}
};


class PortalTriangle : public Triangle{
public:
	PortalTriangle(vec3 a, vec3 b, vec3 c, Material* _material) : Triangle(a, b, c, _material) {

	}
};
Material* material2 = new SmoothMaterial(vec3(1.0f, 1.0f, 1.0f));

void Dodecatriangles(std::vector<Intersectable* >* objects) {
	int ab = 0;
	for (Face face : D.faces) {
	//Face face = D.faces[0];
		vec3 f, g, h, i, j;
		vec3 a = D.vertecies[face.vertecies[0]], b = D.vertecies[face.vertecies[1]], c = D.vertecies[face.vertecies[2]], d = D.vertecies[face.vertecies[3]], e = D.vertecies[face.vertecies[4]];
		f = a+meshWidth * normalize((b - a) + (e - a));
		g = b+meshWidth * normalize((c - b) + (a - b));
		h = c+meshWidth * normalize((d - c) + (b - c)); 
		i = d+meshWidth * normalize((e - d) + (c - d));
		j = e+meshWidth * normalize((a - e) + (d - e));

		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2, 2, 2);
		Material* material = new RoughMaterial(kd1, ks, 50);
		

		vec3 middle = (f + g + h + i + j) / 5.0f;
		D.middlepoints[ab] = middle;
		ab = ab + 1;
		

		objects->push_back(new DiffuseTriangle(a, b, f, material));
		objects->push_back(new DiffuseTriangle(f, b, g, material));
		objects->push_back(new DiffuseTriangle(g, b, c, material));
		objects->push_back(new DiffuseTriangle(g, h, c, material));
		objects->push_back(new DiffuseTriangle(c, d, h, material));
		objects->push_back(new DiffuseTriangle(h, i, d, material));
		objects->push_back(new DiffuseTriangle(d, e, i, material));
		objects->push_back(new DiffuseTriangle(i, j, e, material));
		objects->push_back(new DiffuseTriangle(e, a, j, material));
		objects->push_back(new DiffuseTriangle(j, f, a, material));
		
		objects->push_back(new PortalTriangle(f, g, h, material2));
		objects->push_back(new PortalTriangle(f, i, h, material2));
		objects->push_back(new PortalTriangle(f, i, j, material2));
	}
}

vec3 reflect(vec3 v, vec3 normal) {
	return v - normal * dot(normal, v) * 2;
}


vec3 rotate(vec3 v, vec3 n) {
	mat4 rotmat = RotationMatrix(72.0f, n);
	vec4 vstroke = vec4(v.x, v.y, v.z, 1);
	vstroke = vstroke * rotmat;
	return vec3(vstroke.x, vstroke.y, vstroke.z);
}


vec3 shift(vec3 n, vec3 p) {
	vec3 best = vec3(0, 0, 0);
	float shortest = 0.0f;
	for (const vec3 &middlepoint : D.middlepoints) {
		if (length(p - middlepoint) < shortest) {
			best = middlepoint;
			shortest = length(p - middlepoint);
		}
	}
	vec3 v = (p - best);
	float s = length(v);
	 v = normalize(v);
	return best + s * rotate(v, n);
}



vec3 Fresnel(vec3 F0, float cosTheta) {
	return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
}


template<class T> struct Dnum { 

	float f; 
	T d; 
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};


template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }

typedef Dnum<vec3> Dnum3;

const float A = 0.35f;
const float B = 0.5f;
const float C = 0.1f;
const float R = 0.3f;


vec3 grad(float x, float y, float z) {
	Dnum3 X(x, vec3(1, 0, 0)), Y(y, vec3(0, 1, 0)), Z(z, vec3(0, 0, 1));
	Dnum3 F = Exp(X * X * Dnum3(A, vec3(0, 0, 0)) + Y * Y * Dnum3(B, vec3(0, 0, 0)) - Z * Dnum3(C, vec3(0, 0, 0))) + Dnum3((-1.0f), vec3(0, 0, 0));//lehagyhatjuk a -1 et mert deriváljuk?
	vec3 grad = F.d;
	return grad;
}

vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2, 2, 2);
Material* mat = new RoughMaterial(kd1, ks, 50);
Material* material3 = new RoughMaterial(vec3(0.5, 0.1, 0.1), ks, 50);
class Elipsoid : public Intersectable{
	vec3 center;
	Material* material =  new SmoothMaterial(vec3(kys(0.17, 3.1), kys(0.35, 2.7), kys(1.5, 1.9)));
	float radius = R;
public:
	Elipsoid(vec3 _center) {
		center = _center;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		
		vec3 dist = ray.start - center;
		
		float a = A * ray.dir.x * ray.dir.x + B * ray.dir.y * ray.dir.y;
		float b = 2.0f * A * ray.dir.x * dist.x + 2.0f * B * ray.dir.y * dist.y - C * ray.dir.z;
		float c = A * dist.x * dist.x + B * dist.y * dist.y - C * dist.z - 0.02f;
		float discr = b * b - 4.0f * a * c;
		
		if (discr < 0) 
			return hit;
		
		
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		
		if (dot(hit.position, hit.position) > R * R) {
			hit.position = ray.start + ray.dir * t1;
			if (dot(hit.position, hit.position) > R * R) {
				return intersectC(ray);
			}
		}
		
		hit.normal = normalize(grad(hit.position.x, hit.position.y, hit.position.z)); 

		hit.material = material;
		return hit;
		
	}

	Hit intersectC(const Ray& ray) {
		Hit hit;
		return hit;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat; fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2.0f);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right));
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;

		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};

const Light* light = new Light(vec3(0.5, 0.7, 0.5), vec3(2, 2, 2), vec3(0.4f, 0.4f, 0.4f));
vec3 trace(Ray ray);



const float epsilon = 0.0001f;
const int maxdepth = 5;
float rnd() { return (float)rand() / RAND_MAX; }

float kys(float n, float k) {
	return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k);
}

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, -1.0f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45*2.0f * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le = (2, 2, 2);
		lights.push_back(new Light(lightDirection, Le,La));

		

		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2, 2, 2);
		Material* material1 = new RoughMaterial(kd1, ks, 50);
		Material* material2 = new RoughMaterial(kd2, ks, 50);
		
		vec3 gh(0.17, 0.35, 1.5), hg(3.1, 2.7, 1.9), asd(0.17 / 3.1, 0.35 / 2.7, 1.5 / 1.9);

		Dodecatriangles(&objects);
			
		objects.push_back(new Elipsoid(vec3(0, 0.0f, 0.0f)));
	}

	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp paralel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
		printf("Rendering time: %d miliseconds\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray, float d) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0 && object->intersect(ray).t < d /*&& object->intersect(ray).material != material2*/)return true;
		return false;
	}

	void Animate(float dt) { camera.Animate(dt); }
};

Scene scene;
GPUProgram gpuProgram;

vec3 trace(Ray ray) {
	vec3 weight = vec3(1, 1, 1);
	vec3 outRadiance = vec3(0, 0, 0);
	for (int d = 0; d < maxdepth; d++) {
		Hit hit = scene.firstIntersect(ray);
		if (hit.t < 0) return weight * light->La;
		if (hit.material->rough == 1) {
			float dist = 1.5f / (length(light->direction - hit.position) * length(light->direction - hit.position));
			if (dist < 1) {
				weight = weight * dist;
			}
			outRadiance = outRadiance + weight * hit.material->ka * light->La;
			Ray shadowRay;
			shadowRay.start = hit.position + hit.normal * epsilon;
			shadowRay.dir = normalize(light->direction - hit.position);
			float cosTheta = dot(hit.normal, normalize(light->direction-hit.position));
			if (cosTheta > 0 && !scene.shadowIntersect(shadowRay, length(light->direction - hit.position))) {
				outRadiance = outRadiance + weight * light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + normalize(light->direction - hit.position));
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + weight * light->Le * hit.material->ks * pow(cosDelta, hit.material->shininess);
			}
		}

		if (hit.material->reflective == 1) {
			weight = weight * Fresnel(hit.material->F0, dot(-ray.dir, hit.normal));
			ray.start = hit.position + hit.normal * epsilon;
			
			if (hit.material == material2) {
				ray.start = shift(hit.normal, hit.position) + hit.normal * epsilon;
				ray.dir = normalize(rotate(ray.dir, hit.normal));
			}
			ray.dir = reflect(ray.dir, hit.normal);

		}
		else return outRadiance;
	}
	return outRadiance + weight * light->La;
}

const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	out vec2 texcoord;

	void main() {
		texcoord = (vp + vec2(1, 1))/2;
		gl_Position = vec4(vp.x, vp.y, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	in vec2 texcoord;
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor =texture(textureUnit, texcoord);	// computed color is the color of the primitive
	}
)";


class FullScreenTextureQuad {
	unsigned int vao = 0, textureId = 0;
public:
	FullScreenTextureQuad(int windowWidth, int windowHeight) {

		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;		
		glGenBuffers(1, &vbo);	
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), &vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0 , 4 );

	}
};

FullScreenTextureQuad* fullScreenTextureQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	scene.build();
	fullScreenTextureQuad = new FullScreenTextureQuad(windowWidth, windowHeight);

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTextureQuad->LoadTexture(image);
	fullScreenTextureQuad->Draw();
	glutSwapBuffers();
}


void onKeyboard(unsigned char key, int pX, int pY) {       
}


void onKeyboardUp(unsigned char key, int pX, int pY) {
}


void onMouseMotion(int pX, int pY) {	

}


void onMouse(int button, int state, int pX, int pY) { 

}


void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay();
}

