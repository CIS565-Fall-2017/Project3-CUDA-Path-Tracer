#include "bounds.h"



//Bounds3f Bounds3f::Apply(const glm::mat4 &tr)
//{
//    std::vector<glm::vec3> vertexPos;
//
//
//    vertexPos.push_back(glm::vec3(min.x, min.y, min.z));
//    vertexPos.push_back(glm::vec3(min.x, min.y, max.z));
//    vertexPos.push_back(glm::vec3(min.x, max.y, min.z));
//    vertexPos.push_back(glm::vec3(min.x, max.y, max.z));
//    vertexPos.push_back(glm::vec3(max.x, min.y, min.z));
//    vertexPos.push_back(glm::vec3(max.x, min.y, max.z));
//    vertexPos.push_back(glm::vec3(max.x, max.y, min.z));
//    vertexPos.push_back(glm::vec3(max.x, max.y, max.z));
//
//
//    float new_min_x = FLT_MAX;
//    float new_min_y = FLT_MAX;
//    float new_min_z = FLT_MAX;
//
//    float new_max_x = FLT_MIN;
//    float new_max_y = FLT_MIN;
//    float new_max_z = FLT_MIN;
//
//    for(glm::vec3 each : vertexPos){
//
//		glm::vec3 newVertexPos = glm::vec3(tr * glm::vec4(each, 1.0f));
//
//        // process new X pos
//        if(newVertexPos.x < new_min_x){
//            new_min_x = newVertexPos.x;
//        }
//        if(newVertexPos.x > new_max_x){
//            new_max_x = newVertexPos.x;
//        }
//
//        // process new Y pos
//        if(newVertexPos.y < new_min_y){
//            new_min_y = newVertexPos.y;
//        }
//        if(newVertexPos.y > new_max_y){
//            new_max_y = newVertexPos.y;
//        }
//
//        // process new Z pos
//        if(newVertexPos.z < new_min_z){
//            new_min_z = newVertexPos.z;
//        }
//        if(newVertexPos.z > new_max_z){
//            new_max_z = newVertexPos.z;
//        }
//    }
//
//    return Bounds3f(glm::vec3(new_min_x, new_min_y, new_min_z),
//					glm::vec3(new_max_x, new_max_y, new_max_z));
//}

float Bounds3f::SurfaceArea() const
{
	glm::vec3 d = Diagonal();

    return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
}

Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2)
{
    if(b1.min.x == 0.f && b1.max.x == 0.f &&
       b1.min.y == 0.f && b1.max.y == 0.f &&
       b1.min.z == 0.f && b1.max.z == 0.f){
        return b2;
    }


    else{
        return Bounds3f(glm::vec3(glm::min(b1.min.x, b2.min.x),
								  glm::min(b1.min.y, b2.min.y),
								  glm::min(b1.min.z, b2.min.z)),
						glm::vec3(glm::max(b1.max.x, b2.max.x),
							      glm::max(b1.max.y, b2.max.y),
							      glm::max(b1.max.z, b2.max.z)));
    }
}

Bounds3f Union(const Bounds3f& b1, const glm::vec3& p)
{
    return Bounds3f(glm::vec3(glm::min(b1.min.x, p.x),
							glm::min(b1.min.y, p.y),
							glm::min(b1.min.z, p.z)),
				    glm::vec3(glm::max(b1.max.x, p.x),
							glm::max(b1.max.y, p.y),
							glm::max(b1.max.z, p.z)));
}

Bounds3f Union(const Bounds3f& b1, const glm::vec4& p)
{
    return Union(b1, glm::vec3(p));
}


