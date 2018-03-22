#version 450 core


layout (location =  0) in vec2 TexCoords;
layout (location =  1) in vec3 tanLightDir;
layout (location =  2) in vec3 tanViewDir;


out vec4 FragColor;

// texture samplers
uniform int textureFlags;
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;
uniform sampler2D texture_normal1;
//uniform sampler2D texture_height1;
uniform sampler2D texture_shininess1;
uniform sampler2D texture_emissive1;

const int diffuseBit        = 0;
const int specBit           = 1;
const int normalBit         = 2;
const int heightBit         = 3;
const int shininessBit      = 4;
const int emissiveBit       = 5;


void main() {
    // obtain normal from normal map in range [0,1]
    // transform normal vector to range [-1,1]
    const bool hasDiffuse        = (textureFlags >> diffuseBit & 1)     == 1;
    const bool hasSpecular       = (textureFlags >> specBit & 1)        == 1;
    const bool hasNormal         = (textureFlags >> normalBit & 1)      == 1;
    const bool hasHeight         = (textureFlags >> heightBit & 1)      == 1;
    const bool hasShininess      = (textureFlags >> shininessBit & 1)   == 1;
    const bool hasEmissive       = (textureFlags >> emissiveBit & 1)    == 1;

    const vec4 color = hasDiffuse ? texture(texture_diffuse1, TexCoords) : vec4(1.f);
    const vec3 normal = hasNormal ? normalize( texture(texture_normal1, TexCoords).rgb * 2.f - 1.f) : vec3(0.f, 0.f, 1.f);
    vec3 specMap = hasSpecular ? texture(texture_specular1, TexCoords).rgb : vec3(0.5f);
    const float shininessMap = hasShininess ? texture(texture_shininess1, TexCoords).r : 0.f;
    float specPower = hasShininess ? pow(2.f, 13.f*shininessMap) : 32.f;
    specPower = !hasShininess && hasSpecular ? pow(2.f, 13.f*specMap.r) : specPower;
    specMap = hasShininess && !hasSpecular ? vec3(shininessMap) : specMap;
    const vec3 emissive = hasEmissive ? texture(texture_emissive1, TexCoords).rgb : vec3(0.f);


    // ambient
    vec3 ambient = 0.1 * color.rgb;

    // diffuse
    vec3 lightDir = normalize(tanLightDir);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * color.rgb;

    // specular
    vec3 viewDir = normalize(tanViewDir);

    vec3 halfwayDir = normalize(lightDir + viewDir);

    const float sameHemi = lightDir.z < 0.f || viewDir.z < 0.f ? 0.f : 1.f;
    const float spec = pow(max(dot(normal, halfwayDir), 0.0), specPower) * sameHemi;
    const vec3 specular = specMap * spec;

    //final color
    FragColor = vec4(ambient + diffuse + specular + emissive, color.a);
}
