#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <array>


AABB Apply(const glm::mat4 &tr, AABB paramAABB)
{
	//TODO
	Point3f Points[8];

	Points[0] = glm::vec3(tr * glm::vec4(paramAABB.min.x, paramAABB.min.y, paramAABB.min.z, 1.0f));
	Points[1] = glm::vec3(tr * glm::vec4(paramAABB.min.x, paramAABB.min.y, paramAABB.max.z, 1.0f));

	Points[2] = glm::vec3(tr * glm::vec4(paramAABB.min.x, paramAABB.max.y, paramAABB.min.z, 1.0f));
	Points[3] = glm::vec3(tr * glm::vec4(paramAABB.min.x, paramAABB.max.y, paramAABB.max.z, 1.0f));

	Points[4] = glm::vec3(tr * glm::vec4(paramAABB.max.x, paramAABB.min.y, paramAABB.min.z, 1.0f));
	Points[5] = glm::vec3(tr * glm::vec4(paramAABB.max.x, paramAABB.min.y, paramAABB.max.z, 1.0f));

	Points[6] = glm::vec3(tr * glm::vec4(paramAABB.max.x, paramAABB.max.y, paramAABB.min.z, 1.0f));
	Points[7] = glm::vec3(tr * glm::vec4(paramAABB.max.x, paramAABB.max.y, paramAABB.max.z, 1.0f));


	float maxX = Points[0].x;
	float maxY = Points[0].y;
	float maxZ = Points[0].z;

	float minX = Points[0].x;
	float minY = Points[0].y;
	float minZ = Points[0].z;

	for (int i = 1; i < 8; i++)
	{
		maxX = glm::max(Points[i].x, maxX);
		maxY = glm::max(Points[i].y, maxY);
		maxZ = glm::max(Points[i].z, maxZ);

		minX = glm::min(Points[i].x, minX);
		minY = glm::min(Points[i].y, minY);
		minZ = glm::min(Points[i].z, minZ);
	}


	AABB newBB;

	newBB.min = glm::vec3(minX, minY, minZ);
	newBB.max = glm::vec3(maxX, maxY, maxZ);
	newBB.mid = (newBB.min + newBB.max) * 0.5f;
	return newBB;
}

Vector3f Diagonal(AABB paramAABB)
{
	return paramAABB.max - paramAABB.min;
}

float SurfaceArea(AABB paramAABB)
{
	//TODO
	Vector3f d = Diagonal(paramAABB);
	return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
}

AABB Union(const AABB& b1, const AABB& b2)
{
	AABB newAABB;

	newAABB.min = Point3f(std::min(b1.min.x, b2.min.x), std::min(b1.min.y, b2.min.y), std::min(b1.min.z, b2.min.z));
	newAABB.max = Point3f(std::max(b1.max.x, b2.max.x),	std::max(b1.max.y, b2.max.y), std::max(b1.max.z, b2.max.z));

	return newAABB;
}

AABB Union(const AABB& b1, const glm::vec3& p)
{
	AABB newAABB;
	
	newAABB.min = Point3f(std::min(b1.min.x, p.x), std::min(b1.min.y, p.y), std::min(b1.min.z, p.z));
	newAABB.max = Point3f(std::max(b1.max.x, p.x), std::max(b1.max.y, p.y), std::max(b1.max.z, p.z));

	return newAABB;
}

AABB Union(const AABB& b1, const glm::vec4& p)
{
	return Union(b1, Point3f(p));
}


std::vector<char> readBMP(const std::string &file)
{
	static constexpr size_t HEADER_SIZE = 54;

	std::ifstream bmp(file, std::ios::binary);

	std::array<char, HEADER_SIZE> header;
	bmp.read(header.data(), header.size());

	auto fileSize = *reinterpret_cast<uint32_t *>(&header[2]);
	auto dataOffset = *reinterpret_cast<uint32_t *>(&header[10]);
	auto width = *reinterpret_cast<uint32_t *>(&header[18]);
	auto height = *reinterpret_cast<uint32_t *>(&header[22]);
	auto depth = *reinterpret_cast<uint16_t *>(&header[28]);

	std::cout << "fileSize: " << fileSize << std::endl;
	std::cout << "dataOffset: " << dataOffset << std::endl;
	std::cout << "width: " << width << std::endl;
	std::cout << "height: " << height << std::endl;
	std::cout << "depth: " << depth << "-bit" << std::endl;

	std::vector<char> img(dataOffset - HEADER_SIZE);
	bmp.read(img.data(), img.size());

	auto dataSize = ((width * 3 + 3) & (~3)) * height;
	img.resize(dataSize);
	bmp.read(img.data(), img.size());

	char temp = 0;

	for (auto i = dataSize - 4; i >= 0; i -= 3)
	{
		temp = img[i];
		img[i] = img[i + 2];
		img[i + 2] = temp;

		std::cout << "R: " << int(img[i] & 0xff) << " G: " << int(img[i + 1] & 0xff) << " B: " << int(img[i + 2] & 0xff) << std::endl;
	}

	return img;
}

static short le_short(unsigned char *bytes)
{
	return bytes[0] | ((char)bytes[1] << 8);
}

static void *read_tga(const char *filename, int *width, int *height)
{
	struct tga_header {
		char  id_length;
		char  color_map_type;
		char  data_type_code;
		unsigned char  color_map_origin[2];
		unsigned char  color_map_length[2];
		char  color_map_depth;
		unsigned char  x_origin[2];
		unsigned char  y_origin[2];
		unsigned char  width[2];
		unsigned char  height[2];
		char  bits_per_pixel;
		char  image_descriptor;
	} header;
	int i, color_map_size, pixels_size;
	FILE *f;
	size_t read;
	void *pixels;
	/*errno_t err;
	err = */
	fopen_s(&f, filename, "rb");

	if (!f) {
		fprintf(stderr, "Unable to open %s for reading\n", filename);
		return NULL;
	}

	read = fread(&header, 1, sizeof(header), f);

	if (read != sizeof(header)) {
		fprintf(stderr, "%s has incomplete tga header\n", filename);
		fclose(f);
		return NULL;
	}
	if (header.data_type_code != 2) {
		fprintf(stderr, "%s is not an uncompressed RGB tga file\n", filename);
		fclose(f);
		return NULL;
	}
	/*
	if (header.bits_per_pixel != 24) {
	fprintf(stderr, "%s is not a 24-bit uncompressed RGB tga file\n", filename);
	fclose(f);
	return NULL;
	}
	*/
	for (i = 0; i < header.id_length; ++i)
		if (getc(f) == EOF) {
			fprintf(stderr, "%s has incomplete id string\n", filename);
			fclose(f);
			return NULL;
		}

	color_map_size = le_short(header.color_map_length) * (header.color_map_depth / 8);
	for (i = 0; i < color_map_size; ++i)
		if (getc(f) == EOF) {
			fprintf(stderr, "%s has incomplete color map\n", filename);
			fclose(f);
			return NULL;
		}

	*width = le_short(header.width); *height = le_short(header.height);
	pixels_size = *width * *height * (header.bits_per_pixel / 8);
	pixels = malloc(pixels_size);

	read = fread(pixels, 1, pixels_size, f);
	fclose(f);

	if (read != (unsigned int)pixels_size) {
		fprintf(stderr, "%s has incomplete image\n", filename);
		free(pixels);
		return NULL;
	}

	return pixels;

	fclose(f);
}

void LoadTGA(const char* token_1, std::vector<Image> &Images, std::vector<glm::vec3> &imageData)
{
	DWORD DW = 1024;
	WCHAR FilePath[MAX_PATH];

	GetModuleFileNameW(NULL, FilePath, DW);
	std::wstring source(FilePath);

	size_t lastposition;
	UINT i = 0;
	while (i < 3)
	{
		lastposition = source.rfind(L"\\", source.length());
		source = source.substr(0, lastposition);
		i++;
	}

	char * tempPath = ConvertWCtoC(source.c_str());
	char * newPath = new char[MAX_PATH];
	strcpy(newPath, tempPath);

	delete[] tempPath;

	strcat(newPath, token_1);

	int w;
	int h;

	uint8_t *pixels = (uint8_t *)read_tga(newPath, &w, &h);

	Image imgHeader;
	imgHeader.ID = (int)Images.size();
	imgHeader.width = w;
	imgHeader.height = h;
	imgHeader.beginIndex = (int)imageData.size();

	Images.push_back(imgHeader);


	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			glm::vec3 tt = glm::vec3(pixels[3 * j + 2 + 3 * i*h] / 255.0f, pixels[3 * j + 1 + 3 * i*h] / 255.0f, pixels[3 * j + 3 * i*h] / 255.0f);
			imageData.push_back(tt);
		}
	}

	//std::string temp(newPath);
	//readBMP(temp);

	std::cout << newPath << std::endl;

	delete[] pixels;
	delete[] newPath;
}

char * ConvertWCtoC(const wchar_t* str)
{
	char* pStr;
	int strSize = WideCharToMultiByte(CP_ACP, 0, str, -1, NULL, 0, NULL, NULL);
	pStr = new char[strSize];
	WideCharToMultiByte(CP_ACP, 0, str, -1, pStr, strSize, 0, 0);
	return pStr;
}

bool Inbound(AABB big, AABB sMall)
{
	if (!(big.max.x > sMall.max.x && big.min.x <= sMall.min.x))
	{
		return false;
	}

	if (!(big.max.y > sMall.max.y && big.min.y <= sMall.min.y))
	{
		return false;
	}

	if (!(big.max.z > sMall.max.z && big.min.z <= sMall.min.z))
	{
		return false;
	}

	return true;
}

bool bSpan(AABB big, AABB sMall)
{
	if (big.max.x < sMall.min.x || big.max.y < sMall.min.y || big.max.z < sMall.min.z ||
		big.min.x > sMall.max.x || big.min.y > sMall.max.y || big.min.z > sMall.max.z )
	{
		return false;
	}

	return true;
}


bool sortByX(Triangle t1, Triangle t2)
{
	float t1_max = std::max(std::max(t1.position0.x, t1.position1.x), t1.position2.x);
	float t2_max = std::max(std::max(t2.position0.x, t2.position1.x), t2.position2.x);

	return t1_max < t2_max;
}

bool sortByY(Triangle t1, Triangle t2)
{
	float t1_max = std::max(std::max(t1.position0.y, t1.position1.y), t1.position2.y);
	float t2_max = std::max(std::max(t2.position0.y, t2.position1.y), t2.position2.y);

	return t1_max < t2_max;
}

bool sortByZ(Triangle t1, Triangle t2)
{
	float t1_max = std::max(std::max(t1.position0.z, t1.position1.z), t1.position2.z);
	float t2_max = std::max(std::max(t2.position0.z, t2.position1.z), t2.position2.z);

	return t1_max < t2_max;
}

void createKDtree(std::vector<KDtreeNode> &KDtree, int thisIndex, int MaxTriangles)
{
	KDtreeNode &thisNode = KDtree[thisIndex];
	
	if (KDtree[thisIndex].triangles.size() <= MaxTriangles)
	{
		KDtree[thisIndex].bLeaf = true;
		KDtree[thisIndex].LeftID = -1;
		KDtree[thisIndex].RightID = -1;
		return;
	}

	//Find the widest Axis

	/*
	AABB innerBox = KDtree[thisIndex].triangles[0].boundingBox;
	
	for (int i = 1; i < KDtree[thisIndex].triangles.size(); i++)
	{
		innerBox = Union(innerBox, KDtree[thisIndex].triangles[i].boundingBox);
	}

	
	float x_width = glm::abs(innerBox.max.x - innerBox.min.x);
	float y_width = glm::abs(innerBox.max.y - innerBox.min.y);
	float z_width = glm::abs(innerBox.max.z - innerBox.min.z);
	*/
	
	float x_width = glm::abs(KDtree[thisIndex].boundingBox.max.x - KDtree[thisIndex].boundingBox.min.x);
	float y_width = glm::abs(KDtree[thisIndex].boundingBox.max.y - KDtree[thisIndex].boundingBox.min.y);
	float z_width = glm::abs(KDtree[thisIndex].boundingBox.max.z - KDtree[thisIndex].boundingBox.min.z);	

	int wAxis = -1;

	if (x_width >= y_width && x_width >= z_width)
	{
		wAxis = 0;
	}
	else if (y_width >= x_width && y_width >= z_width)
	{
		wAxis = 1;
	}
	else if (z_width >= x_width && z_width >= y_width)
	{
		wAxis = 2;
	}

	//sort by the axis	
	if (wAxis == 0)
	{
		std::sort(KDtree[thisIndex].triangles.begin(), KDtree[thisIndex].triangles.end(), sortByX);
	}
	else if (wAxis == 1)
	{
		std::sort(KDtree[thisIndex].triangles.begin(), KDtree[thisIndex].triangles.end(), sortByY);
	}
	else if (wAxis == 2)
	{
		std::sort(KDtree[thisIndex].triangles.begin(), KDtree[thisIndex].triangles.end(), sortByZ);
	}
	else
		return;	

	float min_cost = FLT_MAX;
	int min_index = -1;
	AABB min_LeftAABB;
	AABB min_RightAABB;
	int min_Leftcount;
	int min_Rightcount;

	for (int i = 0; i < KDtree[thisIndex].triangles.size(); i++)
	{
		float newBorder;
		AABB LeftAABB = KDtree[thisIndex].boundingBox;
		AABB RightAABB = KDtree[thisIndex].boundingBox;
		
		if (wAxis == 0)
		{
			newBorder = KDtree[thisIndex].triangles[i].boundingBox.max.x;
			LeftAABB.max.x = newBorder;
			RightAABB.min.x = newBorder;
		}
		else if (wAxis == 1)
		{
			newBorder = KDtree[thisIndex].triangles[i].boundingBox.max.y;
			LeftAABB.max.y = newBorder;
			RightAABB.min.y = newBorder;
		}
		else if (wAxis == 2)
		{
			newBorder = KDtree[thisIndex].triangles[i].boundingBox.max.z;
			LeftAABB.max.z = newBorder;
			RightAABB.min.z = newBorder;
		}

		int leftCount = 0;
		int rightCount = 0;

		for (int j = 0; j < KDtree[thisIndex].triangles.size(); j++)
		{
			if (bSpan(LeftAABB, KDtree[thisIndex].triangles[j].boundingBox))
			{
				leftCount++;
			}

			if (bSpan(RightAABB, KDtree[thisIndex].triangles[j].boundingBox))
			{
				rightCount++;
			}
		}

		float cost = (SurfaceArea(LeftAABB) * leftCount + SurfaceArea(RightAABB) * rightCount) / SurfaceArea(KDtree[thisIndex].boundingBox);

		if (min_cost > cost)
		{
			min_cost = cost;
			min_index = i;
			min_LeftAABB = LeftAABB;
			min_RightAABB = RightAABB;
			min_Leftcount = leftCount;
			min_Rightcount = rightCount;
		}
	}

	if (KDtree[thisIndex].triangles.size() == min_Leftcount || KDtree[thisIndex].triangles.size() == min_Rightcount)
	{
		KDtree[thisIndex].bLeaf = true;
		KDtree[thisIndex].LeftID = -1;
		KDtree[thisIndex].RightID = -1;
		return;
	}


	//Divide

	KDtreeNode LNode;
	KDtreeNode RNode;

	LNode.boundingBox = min_LeftAABB;
	RNode.boundingBox = min_RightAABB;

	LNode.ID = KDtree.size();
	LNode.ParentID = KDtree[thisIndex].ID;
	LNode.bLeaf = false;
	KDtree[thisIndex].LeftID = LNode.ID;
	KDtree.push_back(LNode);

	RNode.ID = KDtree.size();
	RNode.ParentID = KDtree[thisIndex].ID;
	RNode.bLeaf = false;
	KDtree[thisIndex].RightID = RNode.ID;
	KDtree.push_back(RNode);

	/*
	for (int j = 0; j < min_Leftcount; j++)
	{
		KDtree[LNode.ID].triangles.push_back(KDtree[thisIndex].triangles[j]);
	}

	for (int j = KDtree[thisIndex].triangles.size() - 1; j >= KDtree[thisIndex].triangles.size() - min_Rightcount - 1; j--)
	{
		KDtree[RNode.ID].triangles.push_back(KDtree[thisIndex].triangles[j]);
	}
	*/
	
	for (int j = 0; j < KDtree[thisIndex].triangles.size(); j++)
	{	
		if (bSpan(min_LeftAABB, KDtree[thisIndex].triangles[j].boundingBox))
		{			
			KDtree[LNode.ID].triangles.push_back(KDtree[thisIndex].triangles[j]);
		}

		if (bSpan(min_RightAABB, KDtree[thisIndex].triangles[j].boundingBox))
		{
			KDtree[RNode.ID].triangles.push_back(KDtree[thisIndex].triangles[j]);
		}
	}
	

	KDtree[LNode.ID].size = KDtree[LNode.ID].triangles.size();
	KDtree[RNode.ID].size = KDtree[RNode.ID].triangles.size();

	KDtree[thisIndex].triangles.clear();

	
	createKDtree(KDtree, LNode.ID, MaxTriangles);
	createKDtree(KDtree, RNode.ID, MaxTriangles);



}

void createBVHtree(std::vector<BVHNode> &BVHtree, int thisIndex, int MaxTriangles)
{
	BVHNode &thisNode = BVHtree[thisIndex];

	if (BVHtree[thisIndex].triangles.size() <= MaxTriangles)
	{
		BVHtree[thisIndex].bLeaf = true;
		BVHtree[thisIndex].LeftID = -1;
		BVHtree[thisIndex].RightID = -1;
		return;
	}

	//Find the widest Axis
	float x_width = glm::abs(thisNode.boundingBox.max.x - thisNode.boundingBox.min.x);
	float y_width = glm::abs(thisNode.boundingBox.max.y - thisNode.boundingBox.min.y);
	float z_width = glm::abs(thisNode.boundingBox.max.z - thisNode.boundingBox.min.z);	

	int wAxis = -1;

	if (x_width >= y_width && x_width >= z_width)
	{
		wAxis = 0;
	}
	else if (y_width >= x_width && y_width >= z_width)
	{
		wAxis = 1;
	}
	else if (z_width >= x_width && z_width >= y_width)
	{
		wAxis = 2;
	}

	//sort by the axis	
	if (wAxis == 0)
	{
		std::sort(BVHtree[thisIndex].triangles.begin(), BVHtree[thisIndex].triangles.end(), sortByX);
	}
	else if (wAxis == 1)
	{
		std::sort(BVHtree[thisIndex].triangles.begin(), BVHtree[thisIndex].triangles.end(), sortByY);
	}
	else if (wAxis == 2)
	{
		std::sort(BVHtree[thisIndex].triangles.begin(), BVHtree[thisIndex].triangles.end(), sortByZ);
	}
	else
		return;

	float min_cost = FLT_MAX;
	int min_index = -1;
	AABB min_LeftAABB;
	AABB min_RightAABB;
	int min_Leftcount;
	int min_Rightcount;

	std::vector<AABB> LeftAABB_container;
	std::vector<AABB> RightAABB_container;

	LeftAABB_container.push_back(BVHtree[thisIndex].triangles[0].boundingBox);
	RightAABB_container.push_back(BVHtree[thisIndex].triangles[BVHtree[thisIndex].triangles.size() - 1].boundingBox);
	
	for (int i = 1; i < BVHtree[thisIndex].triangles.size() - 1; i++)
	{
		LeftAABB_container.push_back(Union(LeftAABB_container[i - 1], BVHtree[thisIndex].triangles[i].boundingBox));
		RightAABB_container.push_back(Union(RightAABB_container[i - 1], BVHtree[thisIndex].triangles[BVHtree[thisIndex].triangles.size() - 1 - i].boundingBox));
	}

	AABB LeftAABB;
	int leftCount = 0;

	AABB RightAABB;
	int rightCount = BVHtree[thisIndex].triangles.size();

	for (int i = 0; i < BVHtree[thisIndex].triangles.size() - 1; i++)
	{
		LeftAABB = LeftAABB_container[i];
		RightAABB = RightAABB_container[BVHtree[thisIndex].triangles.size() - i - 2];

		leftCount++;
		rightCount--;
		/*
		if (i == 0)
		{
			LeftAABB = BVHtree[thisIndex].triangles[0].boundingBox;
		}
		else
			LeftAABB = Union(LeftAABB, BVHtree[thisIndex].triangles[i].boundingBox);

		leftCount++;
		
		AABB RightAABB;
		int rightCount = 0;
		for (int j = i + 1; j < BVHtree[thisIndex].triangles.size(); j++)
		{
			if (j == i + 1)
			{
				RightAABB = BVHtree[thisIndex].triangles[j].boundingBox;
			}
			else
				RightAABB = Union(RightAABB, BVHtree[thisIndex].triangles[j].boundingBox);

			rightCount++;
		}
		*/

		float cost = (SurfaceArea(LeftAABB) * leftCount + SurfaceArea(RightAABB) * rightCount) / SurfaceArea(BVHtree[thisIndex].boundingBox);

		if (min_cost > cost)
		{
			min_cost = cost;
			min_index = i;
			min_LeftAABB = LeftAABB;
			min_RightAABB = RightAABB;
			min_Leftcount = leftCount;
			min_Rightcount = rightCount;
		}
	}

	if (BVHtree[thisIndex].triangles.size() == min_Leftcount || BVHtree[thisIndex].triangles.size() == min_Rightcount)
	{
		BVHtree[thisIndex].bLeaf = true;
		BVHtree[thisIndex].LeftID = -1;
		BVHtree[thisIndex].RightID = -1;
		return;
	}


	//Divide

	BVHNode LNode;
	BVHNode RNode;

	LNode.boundingBox = min_LeftAABB;
	RNode.boundingBox = min_RightAABB;

	LNode.ID = BVHtree.size();
	LNode.ParentID = BVHtree[thisIndex].ID;
	LNode.bLeaf = false;
	BVHtree[thisIndex].LeftID = LNode.ID;
	BVHtree.push_back(LNode);

	RNode.ID = BVHtree.size();
	RNode.ParentID = BVHtree[thisIndex].ID;
	RNode.bLeaf = false;
	BVHtree[thisIndex].RightID = RNode.ID;
	BVHtree.push_back(RNode);

	for (int j = 0; j < BVHtree[thisIndex].triangles.size(); j++)
	{
		if (j <= min_index)
		{
			BVHtree[LNode.ID].triangles.push_back(BVHtree[thisIndex].triangles[j]);
		}
		else
		{
			BVHtree[RNode.ID].triangles.push_back(BVHtree[thisIndex].triangles[j]);
		}
	}
	

	BVHtree[LNode.ID].size = BVHtree[LNode.ID].triangles.size();
	BVHtree[RNode.ID].size = BVHtree[RNode.ID].triangles.size();

	BVHtree[thisIndex].triangles.clear();


	createBVHtree(BVHtree, LNode.ID, MaxTriangles);
	createBVHtree(BVHtree, RNode.ID, MaxTriangles);



}


void createOCtree(std::vector<Octree> *octreeStructure, int thisOctreeIndex, std::vector<Triangle> &triangles, const std::vector<int> &IncludeTriIndices)
{
	std::vector<int> child01_Triangles;
	std::vector<int> child02_Triangles;
	std::vector<int> child03_Triangles;
	std::vector<int> child04_Triangles;
	std::vector<int> child05_Triangles;
	std::vector<int> child06_Triangles;
	std::vector<int> child07_Triangles;
	std::vector<int> child08_Triangles;

	std::vector<int> rest_Triangles;

	glm::vec3 center = ( (*octreeStructure)[thisOctreeIndex].boundingBox.max + (*octreeStructure)[thisOctreeIndex].boundingBox.min) / 2.0f;
	glm::vec3 halfLen = glm::abs((*octreeStructure)[thisOctreeIndex].boundingBox.max - center);

	int Count_01 = 0; int Count_02 = 0; int Count_03 = 0; int Count_04 = 0;
	int Count_05 = 0; int Count_06 = 0; int Count_07 = 0; int Count_08 = 0;

	AABB childBB_01;
	childBB_01.min = (*octreeStructure)[thisOctreeIndex].boundingBox.min;
	childBB_01.max = center;

	AABB childBB_02;
	childBB_02 = childBB_01;
	childBB_02.max.x += halfLen.x;
	childBB_02.min.x += halfLen.x;

	AABB childBB_03;
	childBB_03 = childBB_01;
	childBB_03.min.z += halfLen.z;
	childBB_03.max.z += halfLen.z;

	AABB childBB_04;
	childBB_04 = childBB_03;
	childBB_04.max.x += halfLen.x;
	childBB_04.min.x += halfLen.x;

	AABB childBB_05;
	childBB_05 = childBB_01;
	childBB_05.max.y += halfLen.y;
	childBB_05.min.y += halfLen.y;

	AABB childBB_06;
	childBB_06 = childBB_05;
	childBB_06.max.x += halfLen.x;
	childBB_06.min.x += halfLen.x;

	AABB childBB_07;
	childBB_07 = childBB_05;
	childBB_07.min.z += halfLen.z;
	childBB_07.max.z += halfLen.z;

	AABB childBB_08;
	childBB_08 = childBB_07;
	childBB_08.max.x += halfLen.x;
	childBB_08.min.x += halfLen.x;

	

	//Inspect
	for (int t = 0; t < IncludeTriIndices.size() /*thisOctree.size*/; t++)
	{
		int triangleNum = IncludeTriIndices[t];

		//check if it is child
		if (Inbound(childBB_01, triangles[triangleNum].boundingBox))
		{
			child01_Triangles.push_back(triangleNum);
		}
		else if (Inbound(childBB_02, triangles[triangleNum].boundingBox))
		{
			child02_Triangles.push_back(triangleNum);
		}
		else if (Inbound(childBB_03, triangles[triangleNum].boundingBox))
		{
			child03_Triangles.push_back(triangleNum);
		}
		else if (Inbound(childBB_04, triangles[triangleNum].boundingBox))
		{
			child04_Triangles.push_back(triangleNum);
		}
		else if (Inbound(childBB_05, triangles[triangleNum].boundingBox))
		{
			child05_Triangles.push_back(triangleNum);
		}
		else if (Inbound(childBB_06, triangles[triangleNum].boundingBox))
		{
			child06_Triangles.push_back(triangleNum);
		}
		else if (Inbound(childBB_07, triangles[triangleNum].boundingBox))
		{
			child07_Triangles.push_back(triangleNum);
		}
		else if (Inbound(childBB_08, triangles[triangleNum].boundingBox))
		{
			child08_Triangles.push_back(triangleNum);
		}
		else
		{
			rest_Triangles.push_back(triangleNum);
		}


		//triangleNum = triangles[triangleNum].nextTriangleID;
	}

	//int sum = child01_Triangles.size() + child02_Triangles.size() + child03_Triangles.size() + child04_Triangles.size() + child05_Triangles.size() + child06_Triangles.size() + child07_Triangles.size() + child08_Triangles.size();

	if (rest_Triangles.size() > 0 )
	{
		(*octreeStructure)[thisOctreeIndex].bLeaf = true;

		(*octreeStructure)[thisOctreeIndex].firstElement = rest_Triangles[0];
		
		int tri_index = (*octreeStructure)[thisOctreeIndex].firstElement;
		//Link
		for (int i = 1; i < rest_Triangles.size(); i++)
		{
			//triangles[rest_Triangles[i - 1].triangleID].nextTriangleID = triangles[rest_Triangles[i].triangleID].triangleID;

			triangles[rest_Triangles[i - 1]].nextTriangleID = rest_Triangles[i];
		}

		triangles[rest_Triangles[rest_Triangles.size() - 1]].nextTriangleID = -1;
	}
	
	if (child01_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_01;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child01_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child01_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;

		//tempOctree.bTraversed = false;


		(*octreeStructure).push_back(tempOctree);

		(*octreeStructure)[thisOctreeIndex].child_01 = tempOctree.ID;
		
		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child01_Triangles);
	}
	
	if (child02_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_02;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child02_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child02_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;

		//tempOctree.bTraversed = false;

		(*octreeStructure).push_back(tempOctree);

		(*octreeStructure)[thisOctreeIndex].child_02 = tempOctree.ID;

		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child02_Triangles);
	}
	
	if (child03_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_03;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child03_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child03_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;
		
		//tempOctree.bTraversed = false;
		
		(*octreeStructure).push_back(tempOctree);


		(*octreeStructure)[thisOctreeIndex].child_03 = tempOctree.ID;

		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child03_Triangles);
	}
	
	if (child04_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_04;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child04_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child04_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;

		//tempOctree.bTraversed = false;
		(*octreeStructure).push_back(tempOctree);

		(*octreeStructure)[thisOctreeIndex].child_04 = tempOctree.ID;

		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child04_Triangles);
	}
	
	if (child05_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_05;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child05_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child05_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;
		//tempOctree.bTraversed = false;
		(*octreeStructure).push_back(tempOctree);

		(*octreeStructure)[thisOctreeIndex].child_05 = tempOctree.ID;

		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child05_Triangles);
	}
	
	if (child06_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_06;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child06_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child06_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;
		//tempOctree.bTraversed = false;
		(*octreeStructure).push_back(tempOctree);

		(*octreeStructure)[thisOctreeIndex].child_06 = tempOctree.ID;

		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child06_Triangles);
	}
	
	if (child07_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_07;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child07_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child07_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;
		//tempOctree.bTraversed = false;
		(*octreeStructure).push_back(tempOctree);

		(*octreeStructure)[thisOctreeIndex].child_07 = tempOctree.ID;

		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child07_Triangles);
	}
	
	if (child08_Triangles.size() > 0)
	{
		Octree tempOctree;
		tempOctree.ID = (int)(*octreeStructure).size();
		tempOctree.boundingBox = childBB_08;
		tempOctree.bLeaf = false;
		tempOctree.size = (int)child08_Triangles.size();

		tempOctree.child_01 = -1;
		tempOctree.child_02 = -1;
		tempOctree.child_03 = -1;
		tempOctree.child_04 = -1;
		tempOctree.child_05 = -1;
		tempOctree.child_06 = -1;
		tempOctree.child_07 = -1;
		tempOctree.child_08 = -1;
		tempOctree.firstElement = child08_Triangles[0];
		tempOctree.ParentID = thisOctreeIndex;
		//tempOctree.bTraversed = false;
		(*octreeStructure).push_back(tempOctree);

		(*octreeStructure)[thisOctreeIndex].child_08 = tempOctree.ID;

		createOCtree(octreeStructure, (int)(*octreeStructure).size() - 1, triangles, child08_Triangles);
	}
}

float RoughnessToAlpha(float roughness)
{
	roughness = std::max(roughness, (float)1e-3);
	float x = std::log(roughness);
	return 1.62142f + 0.819955f * x + 0.1734f * x * x +
		0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}


Scene::Scene(std::string filename) {
	std::cout << "Reading scene from " << filename << " ..." << std::endl;
	std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
		std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }
    while (fp_in.good()) {
		std::string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
			std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
				std::cout << " " << std::endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
				std::cout << " " << std::endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
				std::cout << " " << std::endl;
			}

        }
    }
}

int Scene::loadGeom(std::string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
		std::cout << "ERROR: OBJECT ID does not match expected number of geoms" << std::endl;
        return -1;
    } else {
		std::cout << "Loading Geom " << id << "..." << std::endl;
        Geom newGeom;
		std::string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
				std::cout << "Creating new sphere..." << std::endl;
                newGeom.type = SPHERE;
            }
			else if (strcmp(line.c_str(), "cube") == 0) {
				std::cout << "Creating new cube..." << std::endl;
                newGeom.type = CUBE;
            }
			else if (strcmp(line.c_str(), "plane") == 0) {
				std::cout << "Creating new plane..." << std::endl;
				newGeom.type = PLANE;
			}
			else if (strcmp(line.c_str(), "mesh") == 0)
			{
				std::cout << "Creating new mesh..." << std::endl;
				newGeom.type = MESH;

				utilityCore::safeGetline(fp_in, line);
				if (!line.empty() && fp_in.good())
				{
					std::vector<tinyobj::shape_t> shapes;
					std::vector<tinyobj::material_t> _materials;
					//tinyobj::attrib_t att;

					DWORD DW = 1024;
					WCHAR FilePath[MAX_PATH];

					GetModuleFileNameW(NULL, FilePath, DW);
					std::wstring source(FilePath);

					size_t lastposition;
					UINT i = 0;
					while (i < 3)
					{
						lastposition = source.rfind(L"\\", source.length());
						source = source.substr(0, lastposition);
						i++;
					}

					char * tempPath = ConvertWCtoC(source.c_str());
					char * newPath = new char[MAX_PATH];
					strcpy(newPath, tempPath);

					delete[] tempPath;

					strcat(newPath, line.c_str());

					std::cout << newPath << std::endl;

					

					std::string errors = tinyobj::LoadObj(shapes, _materials, newPath);


					if (errors.size() == 0)
					{
						int min_idx = 0;
						//Read the information from the vector of shape_ts
						for (unsigned int i = 0; i < shapes.size(); i++)
						{
							std::vector<unsigned int> indices = shapes[i].mesh.indices;
							int tricount = indices.size() / 3;

							std::vector<glm::vec3> Vpositions;
							std::vector<glm::vec3> Vnormals;
							std::vector<glm::vec2> Vuvs;

							Triangle* triangles = new Triangle[tricount];

							std::vector<float> &positions = shapes[i].mesh.positions;
							std::vector<float> &normals = shapes[i].mesh.normals;
							std::vector<float> &uvs = shapes[i].mesh.texcoords;

							for (unsigned int j = 0; j < positions.size()/3; j++)
							{
								Vpositions.push_back(glm::vec3(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]));
							}

							for (unsigned int j = 0; j < normals.size() / 3; j++)
							{
								Vnormals.push_back(glm::vec3(normals[j * 3], normals[j * 3 + 1], normals[j * 3 + 2]));
							}

							for (unsigned int j = 0; j < uvs.size() / 2; j++)
							{
								Vuvs.push_back(glm::vec2(uvs[j * 2], uvs[j * 2 + 1]));
							}

							//

							for (int j = 0; j < tricount; j++)
							{
								int index01 = indices[j * 3];
								int index02 = indices[j * 3 + 1];
								int index03 = indices[j * 3 + 2];

								triangles[j].position0 = Vpositions[index01];
								triangles[j].position1 = Vpositions[index02];
								triangles[j].position2 = Vpositions[index03];

								if (normals.size() == 0)
								{
									glm::vec3 vec1 = triangles[j].position2 - triangles[j].position0;
									glm::vec3 vec2 = triangles[j].position1 - triangles[j].position0;

									triangles[j].normal0 = glm::normalize(glm::cross(vec2, vec1));
									triangles[j].normal1 = triangles[j].normal0;
									triangles[j].normal2 = triangles[j].normal0;
								}
								else
								{
									triangles[j].normal0 = Vnormals[index01];
									triangles[j].normal1 = Vnormals[index02];
									triangles[j].normal2 = Vnormals[index03];
								}
								

								if (uvs.size() == 0)
								{
									triangles[j].texcoord0 = glm::vec2(0.0f, 0.0f);
									triangles[j].texcoord1 = glm::vec2(0.0f, 0.0f);
									triangles[j].texcoord2 = glm::vec2(0.0f, 0.0f);
								}
								else
								{
									triangles[j].texcoord0 = Vuvs[index01];
									triangles[j].texcoord1 = Vuvs[index02];
									triangles[j].texcoord2 = Vuvs[index03];
								}

								
							}


							//link material
							utilityCore::safeGetline(fp_in, line);
							if (!line.empty() && fp_in.good()) {
								std::vector<std::string> tokens = utilityCore::tokenizeString(line);
								newGeom.materialid = atoi(tokens[1].c_str());
								std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << std::endl;
							}

							//load transformations
							utilityCore::safeGetline(fp_in, line);
							while (!line.empty() && fp_in.good()) {
								std::vector<std::string> tokens = utilityCore::tokenizeString(line);

								//load tranformations
								if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
									newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
								}
								else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
									newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
								}
								else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
									newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
								}

								utilityCore::safeGetline(fp_in, line);
							}

							newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale, newGeom.rotationMat);
							newGeom.inverseTransform = glm::inverse(newGeom.transform);
							newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

							newGeom.meshInfo.size = tricount;
							newGeom.meshInfo.triangleBeginIndex = (int)trianglesInMesh.size();
							
							

							for (int z = 0; z < tricount; z++)
							{
								Triangle temp;
								

								triangles[z].planeNormal = glm::normalize(glm::cross(triangles[z].position1 - triangles[z].position0, triangles[z].position2 - triangles[z].position1));
								triangles[z].triangleID = (int)trianglesInMesh.size();

								//for AABB
								glm::vec3 Points_0 = glm::vec3(newGeom.transform * glm::vec4(triangles[z].position0, 1.0f));
								glm::vec3 Points_1 = glm::vec3(newGeom.transform * glm::vec4(triangles[z].position1, 1.0f));
								glm::vec3 Points_2 = glm::vec3(newGeom.transform * glm::vec4(triangles[z].position2, 1.0f));

								float maxX = Points_0.x;
								float maxY = Points_0.y;
								float maxZ = Points_0.z;

								float minX = Points_0.x;
								float minY = Points_0.y;
								float minZ = Points_0.z;

								maxX = glm::max(Points_1.x, maxX);
								maxY = glm::max(Points_1.y, maxY);
								maxZ = glm::max(Points_1.z, maxZ);

								minX = glm::min(Points_1.x, minX);
								minY = glm::min(Points_1.y, minY);
								minZ = glm::min(Points_1.z, minZ);

								maxX = glm::max(Points_2.x, maxX);
								maxY = glm::max(Points_2.y, maxY);
								maxZ = glm::max(Points_2.z, maxZ);

								minX = glm::min(Points_2.x, minX);
								minY = glm::min(Points_2.y, minY);
								minZ = glm::min(Points_2.z, minZ);
								

								triangles[z].boundingBox.min = glm::vec3(minX, minY, minZ) - glm::vec3(0.000001f);
								triangles[z].boundingBox.max = glm::vec3(maxX, maxY, maxZ) + glm::vec3(0.000001f);;
								triangles[z].boundingBox.mid = (triangles[z].boundingBox.min + triangles[z].boundingBox.max) * 0.5f;

								temp = triangles[z];

								if (z == tricount - 1)
									temp.nextTriangleID = -1;
								else
									temp.nextTriangleID = temp.triangleID + 1;

								trianglesInMesh.push_back(temp);
							}
							
							float maxX = triangles[0].boundingBox.max.x;
							float maxY = triangles[0].boundingBox.max.y;
							float maxZ = triangles[0].boundingBox.max.z;

							float minX = triangles[0].boundingBox.min.x;
							float minY = triangles[0].boundingBox.min.y;
							float minZ = triangles[0].boundingBox.min.z;

							for (int z = 1; z < tricount; z++)
							{

								maxX = glm::max(triangles[z].boundingBox.max.x, maxX);
								maxY = glm::max(triangles[z].boundingBox.max.y, maxY);
								maxZ = glm::max(triangles[z].boundingBox.max.z, maxZ);

								minX = glm::min(triangles[z].boundingBox.min.x, minX);
								minY = glm::min(triangles[z].boundingBox.min.y, minY);
								minZ = glm::min(triangles[z].boundingBox.min.z, minZ);

							}

							newGeom.boundingBox.min = glm::vec3(minX, minY, minZ);
							newGeom.boundingBox.max = glm::vec3(maxX, maxY, maxZ);

							newGeom.ID = (int)geoms.size();

							//BVH
							newGeom.meshInfo.BVHtreeID = bvhTree.size();

							BVHNode root;
							root.ID = bvhTree.size();
							root.ParentID = -1;							
							root.size = tricount;

							root.bLeaf = false;
							root.boundingBox = newGeom.boundingBox;

							for (int z = 0; z < tricount; z++)
							{
								root.triangles.push_back(triangles[z]);
							}

							bvhTree.push_back(root);

							createBVHtree(bvhTree, root.ID, 4);

							for (int h = root.ID; h < bvhTree.size(); h++)
							{
								BVHNodeForGPU temp;

								temp.bLeaf = bvhTree[h].bLeaf;
								temp.boundingBox = bvhTree[h].boundingBox;
								temp.ID = bvhTree[h].ID;
								temp.LeftID = bvhTree[h].LeftID;
								temp.RightID = bvhTree[h].RightID;
								temp.ParentID = bvhTree[h].ParentID;
								temp.size = bvhTree[h].size;

								temp.bLeftTraversed = false;
								temp.bRightTraversed = false;

								temp.bLeftIntersected = false;
								temp.bRightIntersected = false;

								temp.minT = FLT_MAX;
								temp.maxT = -FLT_MAX;

								if (temp.bLeaf == true)
								{
									temp.TriangleArrayIndex = bvhTreeTriangleIndexForGPU.size();
									for (int k = 0; k < bvhTree[h].triangles.size(); k++)
									{
										bvhTreeTriangleIndexForGPU.push_back(bvhTree[h].triangles[k].triangleID);
									}
								}

								bvhTreeForGPU.push_back(temp);
							}

							/*
							newGeom.meshInfo.KDtreeID = kdTree.size();

							KDtreeNode root;
							root.ID = kdTree.size();
							root.ParentID = -1;							
							root.size = tricount;

							root.bLeaf = false;
							root.boundingBox = newGeom.boundingBox;

							for (int z = 0; z < tricount; z++)
							{
								root.triangles.push_back(triangles[z]);
							}



							kdTree.push_back(root);
														
							createKDtree(kdTree, root.ID, 4);

							for (int h = root.ID; h < kdTree.size(); h++)
							{
								KDtreeNodeForGPU temp;

								temp.bLeaf = kdTree[h].bLeaf;
								temp.boundingBox = kdTree[h].boundingBox;
								temp.ID = kdTree[h].ID;
								temp.LeftID = kdTree[h].LeftID;
								temp.RightID = kdTree[h].RightID;
								temp.ParentID = kdTree[h].ParentID;
								temp.size = kdTree[h].size;


								temp.bLeftTraversed = false;
								temp.bRightTraversed = false;

								temp.bLeftIntersected = false;
								temp.bRightIntersected = false;

								temp.minT = FLT_MAX;
								temp.maxT = -FLT_MAX;

								if (temp.bLeaf == true)
								{
									temp.TriangleArrayIndex = kdTreeTriangleIndexForGPU.size();
									for (int k = 0; k < kdTree[h].triangles.size(); k++)
									{
										kdTreeTriangleIndexForGPU.push_back(kdTree[h].triangles[k].triangleID);
									}
								}

								kdTreeForGPU.push_back(temp);
							}
							*/



							geoms.push_back(newGeom);

							//This is light
							if (materials[newGeom.materialid].emittance > 0.0)
							{
								lights.push_back(newGeom);
							}

							delete[] triangles;
							
						}						
					}

					delete[] newPath;
				 }
				 return 1;
			}
		}

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
			std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
			std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << std::endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
			std::vector<std::string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale, newGeom.rotationMat);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);		

		


		if (newGeom.type == SPHERE)
		{
			newGeom.boundingBox.min = glm::vec3(-0.5f, -0.5f, -0.5f);
			newGeom.boundingBox.max = glm::vec3(0.5f, 0.5f, 0.5f);

			newGeom.boundingBox = Apply(newGeom.transform, newGeom.boundingBox);

		}
		else if (newGeom.type == CUBE)
		{
			newGeom.boundingBox.min = glm::vec3(-0.5f, -0.5f, -0.5f);
			newGeom.boundingBox.max = glm::vec3(0.5f, 0.5f, 0.5f);

			newGeom.boundingBox = Apply(newGeom.transform, newGeom.boundingBox);
		}
		else if (newGeom.type == PLANE)
		{
			newGeom.boundingBox.min = glm::vec3(-0.5f, -0.5f, -0.001f);
			newGeom.boundingBox.max = glm::vec3(0.5f, 0.5f, 0.001f);

			newGeom.boundingBox = Apply(newGeom.transform, newGeom.boundingBox);
		}


		newGeom.ID = (int)geoms.size();

        geoms.push_back(newGeom);
		

		//This is light
		if (materials[newGeom.materialid].emittance > 0.0)
		{
			lights.push_back(newGeom);
		}

        return 1;
    }
}

int Scene::loadCamera() {
	std::cout << "Loading Camera ..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
	camera.lensRadious = 0.0f;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
		std::string line;
        utilityCore::safeGetline(fp_in, line);
		std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

	std::string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
		std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "FOCALDST") == 0) {
			camera.focalDistance = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "LENSRADIOUS") == 0) {
			camera.lensRadious = atof(tokens[1].c_str());
		}
		
		else if (strcmp(tokens[0].c_str(), "ENVMAP") == 0)
		{
			if (tokens.size() > 1)
			{
				LoadTGA(tokens[1].c_str(), Images, imageData);
				envMapId = (int)Images.size() - 1;
			}
			else
				envMapId = -1;

		}
		
        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

	std::cout << "Loaded camera!" << std::endl;
    return 1;
}

int Scene::loadMaterial(std::string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
		std::cout << "ERROR: MATERIAL ID does not match expected number of materials" << std::endl;
        return -1;
    } else {
		std::cout << "Loading Material " << id << "..." << std::endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 11; i++) {
			std::string line;
            utilityCore::safeGetline(fp_in, line);
			std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            }
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            }
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;            
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            }
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            }
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "DIFFUSEMAP") == 0)
			{
				if (tokens.size() > 1)
				{
					LoadTGA(tokens[1].c_str(), Images, imageData);
					newMaterial.diffuseTexID = (int)Images.size() - 1;

				}
				else
					newMaterial.diffuseTexID = -1;
			}
			else if (strcmp(tokens[0].c_str(), "SPECULAMAP") == 0)
			{
				if (tokens.size() > 1)
				{
					LoadTGA(tokens[1].c_str(), Images, imageData);
					newMaterial.specularTexID = (int)Images.size() - 1;
				}
				else
					newMaterial.specularTexID = -1;
			}
			else if (strcmp(tokens[0].c_str(), "NORMALMAP") == 0)
			{
				if (tokens.size() > 1)
				{
					LoadTGA(tokens[1].c_str(), Images, imageData);
					newMaterial.normalTexID = (int)Images.size() - 1;
				}
				else
					newMaterial.normalTexID = -1;
			
			}
			else if (strcmp(tokens[0].c_str(), "ROUGHNESSMAP") == 0)
			{
				if (tokens.size() > 1)
				{
					LoadTGA(tokens[1].c_str(), Images, imageData);
					newMaterial.roughnessTexID = (int)Images.size() - 1;
				}
				else
					newMaterial.roughnessTexID = -1;
				
			}
        }

		// RoughnessToAlpha(1.0f - newMaterial.hasReflective);
		newMaterial.Roughness = 1.0f - newMaterial.hasReflective;
        materials.push_back(newMaterial);	

        return 1;
    }
}
