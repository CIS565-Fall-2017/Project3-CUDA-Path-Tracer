#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 aTangent;


layout (location =  0) out vec2 TexCoords;
layout (location =  1) out vec3 tanLightDir;
layout (location =  2) out vec3 tanViewDir;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 cameraPosition;

const vec3 worldLightPos = vec3(100.f, 100.f, 100.f);
//const vec3 worldLightPos = vec3(-1.75f, 0.409f, 0.318f);
void main() {
    vec4 worldPos = model * vec4(aPos, 1.f);

    //outputs
	gl_Position = projection * view * worldPos;

    mat3 modelToWorld = transpose(inverse(mat3(model)));
	vec3 worldnor = normalize(modelToWorld * aNormal);
	vec3 worldtan = normalize(modelToWorld * vec3(aTangent));
    worldtan = normalize(worldtan - dot(worldtan, worldnor)*worldnor);
	vec3 worldbitan = aTangent.w * cross(worldnor, worldtan);//w=1 if it was righthanded, -1 left
    mat3 worldToTan = transpose(mat3(worldtan, worldbitan, worldnor));    

    vec3 worldViewDir = vec3(cameraPosition - worldPos);//interp of the normalized would be incorrect
    vec3 worldLightDir = worldLightPos - vec3(worldPos);//interp of teh normalized would be incorrect(unless directional)

    //world to tan space
    tanLightDir = worldToTan * worldLightDir;
    tanViewDir  = worldToTan * worldViewDir;

	TexCoords = aTexCoords;
}
