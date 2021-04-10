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


//unsigned int vao;	   // virtual world on the GPU



/*struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};*/

//---------------------------
struct Material {
	//---------------------------
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};

float kys(float n, float k);

//---------------------------
struct RoughMaterial : Material {
	//---------------------------
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

//---------------------------
struct SmoothMaterial : Material {
	//---------------------------
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

//question whats ka - ambiens kd - diffúz ks - spekuláris visszaverõdés - mit jelentenek ezek és hol kellene használni?

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

//Geomery part

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



//End of geometry part

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
		hit.normal = (hit.position - center) / radius; //minden geometriára külön kell megoldani
		hit.material = material;
		return hit;
	}

};
// My stuff---------------------------------------------------------------------------
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
		//valami rossz a normálvektorral
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
//-------------------------------------------------------------------------------------------------
Material* material2 = new SmoothMaterial(vec3(1.0f, 1.0f, 1.0f));
//-------------------------------------------------------------------------------------------------
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
		
		/*Material* material = new Material(a, b, 12.0f);
		Material* material2 = new Material(a, b, 12.0f);*/

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

//vec3 rotate(vec3 n, vec3 v) {

	/*//the 3D Descates coordinate system vectors
	vec4 x = vec4(1, 0, 0, 0);
	vec4 y = vec4(0, 1, 0, 0);
	vec4 z = vec4(0, 0, 1, 0);

	//counting the angle between n and x in the x,y plane
	vec4 alpha = vec4(n.x, n.y, 0, 0);
	float alphalength = n.x * n.x + n.y * n.y;
	alpha = alpha / alphalength;

	float THETA = acosf(dot(alpha, x));
	
	//rotation matrix
	mat4 rmTHETA = mat4(
		cosf(THETA),   -sinf(THETA),	0.0f,	0.0f,
		sinf(THETA),	cosf(THETA),	0.0f,	0.0f,
		0.0f,			0.0f,			1.0f,	0.0f,
		0.0f,			0.0f,			0.0f,	0.0f);
	
	//rotating n to the x,z plane
	alpha = alpha * rmTHETA;

	//counting the angle between n and z in the x,z plane
	vec4 beta = vec4(alpha.x * alphalength, 0, n.z, 0);

	float betalength = beta.x * beta.x + beta.z * beta.z;
	beta = beta / betalength;

	float SIGMA = acosf(dot(beta, z));

	mat4 rmSIGMA = mat4(
		cosf(THETA),		0.0f,		sinf(THETA),	0.0f,
		0.0f,				1.0f,		0.0f,			0.0f,
		-sinf(THETA),		0.0f,		cosf(THETA),	0.0f,
		0.0f,				0.0f,		0.0f,			0.0f
	);

	//if debugging here beta should be equal to z (0,0,1,0)
	beta = beta * rmSIGMA;

	//now because it is the normal vector that we rotated we can rotate around it by 72 degrees

	float PHI = 72.0f;

	mat4 rmPHI = mat4(
		cosf(PHI),		-sinf(PHI),			0.0f,	0.0f,
		sinf(PHI),		cosf(PHI),			0.0f,	0.0f,
		0.0f,			0.0f,				1.0f,	0.0f,
		0.0f,			0.0f,				0.0f,	0.0f);

		*/

//}

/*vec3 translate(vec3 v, vec3 n) {
	vec3 vsin = v - n * dot(n, v);
	vec3 vstroke = cross(vsin, n);
	return cosf(72.0f) * vsin + sinf(72.0f) * vstroke;
}
*/

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

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
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

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}
/*template<class T> Dnum<T> Exp(Dnum<T> g) {
	return Dnum<T>(expf(g.f), expf(g.f) * g.d);
}*/

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
		//this ones bad
		
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
		
		/*if (!(center.x - R < hit.position.x   && hit.position.x < center.x + R )) {
			return intersectC(ray);
		}else
		if (!(center.y - R < hit.position.y && hit.position.y < center.y + R)) {
			return intersectC(ray);
		}else
		
		if (!(center.z - R < hit.position.z && hit.position.z < center.z + R)) {
			return intersectC(ray);
		}*/
		if (dot(hit.position, hit.position) > R * R) {
			hit.position = ray.start + ray.dir * t1;
			if (dot(hit.position, hit.position) > R * R) {
				return intersectC(ray);
			}
		}
		
		hit.normal = normalize(grad(hit.position.x, hit.position.y, hit.position.z)); //minden geometriára külön kell megoldani
		/*if (ray.start.z < hit.position.z) {
			hit.normal = hit.normal * -1.0f;
		}*/
		hit.material = material;
		return hit;
		
	}

	Hit intersectC(const Ray& ray) {
		Hit hit;
		return hit;
	}
};



//end-------------------------------------------------------------------

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
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;//---------------------------
		//eye = vec3(d.x * cos(dt) + d.y * sin(dt),-d.x * sin(dt) + d.y * cos(dt), d.z ) + lookat;//---------------------------
		//eye = vec3(d.x, d.y * cos(dt) + d.z * sin(dt), -d.y * sin(dt) + d.z * cos(dt) ) + lookat;//---------------------------
		set(eye, lookat, up, fov);
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};

//const Light* light = new Light(vec3(0.5, 0.5, 0.5), vec3(2, 2, 2), vec3(0.4f, 0.4f, 0.4f));
const Light* light = new Light(vec3(0.5, 0.7, 0.5), vec3(2, 2, 2), vec3(0.4f, 0.4f, 0.4f));
vec3 trace(Ray ray);



const float epsilon = 0.0001f;// 0.0001f;
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
		//Material* material2 = new SmoothMaterial(vec3(kys(0.17 , 3.1), kys(0.35 , 2.7), kys(1.5 , 1.9)));

		Dodecatriangles(&objects);
	
		for (int i = 0; i < 2; i++) {
			//objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd()  - 0.5f, rnd()  - 0.5f), rnd() * 0.1f, material1));
			//objects.push_back(new Sphere(vec3(0,0,0), 0.1f, material2));
		}

		for (const vec3& middle : D.middlepoints) {
			//objects.push_back(new Sphere(middle, 0.1f, material2));
		}
			
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
	/*vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		//------------------------------------------------------------------
		/*if (hit.material == material2 && depth > 0) {
			depth--;
			vec3 r = reflect(ray.dir, hit.normal);
			r = translate(r, hit.normal);
			r = normalize(r);
			ray.dir = r;
			ray.start = hit.position;
			hit = firstIntersect(ray);
		}*/
		//------------------------------------------------------------------
		/*if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal  * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}*/
	void Animate(float dt) { camera.Animate(dt); }
};

Scene scene;
GPUProgram gpuProgram;

vec3 trace(Ray ray) {
	vec3 weight = vec3(1, 1, 1);
	vec3 outRadiance = vec3(0, 0, 0);
	for (int d = 0; d < maxdepth; d++) {
		Hit hit = scene.firstIntersect(ray);
		if (hit.t < 0) return weight * light->La;//Le
		if (hit.material->rough == 1) {
			float dist = 1.5f / (length(light->direction - hit.position) * length(light->direction - hit.position));
			if (dist < 1) {
				weight = weight * dist;
			}
			outRadiance = outRadiance + weight * hit.material->ka * light->La;//Le
			Ray shadowRay;
			shadowRay.start = hit.position + hit.normal * epsilon;
			shadowRay.dir = normalize(light->direction - hit.position);//light->direction
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
			
			if (hit.material == material2) {//rossz a translate
				ray.start = shift(hit.normal, hit.position) + hit.normal * epsilon;
				ray.dir = normalize(rotate(ray.dir, hit.normal));
			}
			ray.dir = reflect(ray.dir, hit.normal);

		}
		else return outRadiance;
	}
	return outRadiance + weight * light->La;
}

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
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

// fragment shader in GLSL
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

		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
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

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	/*vec3 r = reflect(vec3(-0.70710678f,0.0f, -0.70710678f), vec3(0,0,1));
	r = translate(r, vec3(0, 0, 1));
	r = normalize(r);*/

	scene.build();
	fullScreenTextureQuad = new FullScreenTextureQuad(windowWidth, windowHeight);
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTextureQuad->LoadTexture(image);
	fullScreenTextureQuad->Draw();

	// Set color to (0, 1, 0) = green
	
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	//	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	scene.Animate(0.1f);
	glutPostRedisplay();
}
/*

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 450
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 450
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Sphere {
		vec3 center;
		float radius;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// material index
	};

	struct Ray {
		vec3 start, dir;
	};

	const int nMaxObjects = 500;

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[2];  // diffuse, specular, ambient ref
	uniform int nObjects;
	uniform Sphere objects[nMaxObjects];

	in  vec3 p;					// point on camera window corresponding to the pixel
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	Hit intersect(const Sphere object, const Ray ray) {
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - object.center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - object.radius * object.radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - object.center) / object.radius;
		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		for (int o = 0; o < nObjects; o++) {
			Hit hit = intersect(objects[o], ray); //  hit.t < 0 if no intersection
			if (o < nObjects/2) hit.mat = 0;	 // half of the objects are rough
			else			    hit.mat = 1;     // half of the objects are reflective
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (int o = 0; o < nObjects; o++) if (intersect(objects[o], ray).t > 0) return true; //  hit.t < 0 if no intersection
		return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 5;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * light.La;
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			} else return outRadiance;
		}
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

//---------------------------
struct Material {
	//---------------------------
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};

//---------------------------
struct RoughMaterial : Material {
	//---------------------------
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

//---------------------------
struct SmoothMaterial : Material {
	//---------------------------
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

//---------------------------
struct Sphere {
	//---------------------------
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius) { center = _center; radius = _radius; }
};

//---------------------------
struct Camera {
	//---------------------------
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	void setUniformMaterials(const std::vector<Material*>& materials) {
		char name[256];
		for (unsigned int mat = 0; mat < materials.size(); mat++) {
			sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name);
			sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
			sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
			sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
			sprintf(name, "materials[%d].F0", mat); setUniform(materials[mat]->F0, name);
			sprintf(name, "materials[%d].rough", mat); setUniform(materials[mat]->rough, name);
			sprintf(name, "materials[%d].reflective", mat); setUniform(materials[mat]->reflective, name);
		}
	}

	void setUniformLight(Light* light) {
		setUniform(light->La, "light.La");
		setUniform(light->Le, "light.Le");
		setUniform(light->direction, "light.direction");
	}

	void setUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}

	void setUniformObjects(const std::vector<Sphere*>& objects) {
		setUniform((int)objects.size(), "nObjects");
		char name[256];
		for (unsigned int o = 0; o < objects.size(); o++) {
			sprintf(name, "objects[%d].center", o);  setUniform(objects[o]->center, name);
			sprintf(name, "objects[%d].radius", o);  setUniform(objects[o]->radius, name);
		}
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

//---------------------------
class Scene {
	//---------------------------
	std::vector<Sphere*> objects;
	std::vector<Light*> lights;
	Camera camera;
	std::vector<Material*> materials;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(1, 1, 1), vec3(3, 3, 3), vec3(0.4f, 0.3f, 0.3f)));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(10, 10, 10);
		materials.push_back(new RoughMaterial(kd, ks, 50));
		materials.push_back(new SmoothMaterial(vec3(0.9f, 0.85f, 0.8f)));

		for (int i = 0; i < 5; i++)
			objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f));

	}

	void setUniform(Shader& shader) {
		shader.setUniformObjects(objects);
		shader.setUniformMaterials(materials);
		shader.setUniformLight(lights[0]);
		shader.setUniformCamera(camera);
	}

	void Animate(float dt) { camera.Animate(dt); }
};

Shader shader; // vertex and fragment shaders
Scene scene;

//---------------------------
class FullScreenTexturedQuad {
	//---------------------------
	unsigned int vao = 0;	// vertex array object id and texture id
public:
	void create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.create();

	// create program for the GPU
	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
}

// Window has become invalid: Redraw
void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("%d msec\r", (tEnd - tStart) / nFrames);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	scene.setUniform(shader);
	fullScreenTexturedQuad.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.00001f);
	glutPostRedisplay();
}*/