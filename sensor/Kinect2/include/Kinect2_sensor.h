//#define KINECT_COLOR_WIDTH		1920
//#define KINECT_COLOR_HEIGHT		1080
//#define KINECT_DEPTH_WIDTH		512
//#define KINECT_DEPTH_HEIGHT		424
//#define NUM_KINECTS				1
//
//#define OPENCV_WAIT_DELAY		10
//
//#define BODY_COUNT				6
//
//#define PI						3.141592653589

//using namespace cv;




//Single Body Structure;
typedef struct BodyInfo{
	Joint JointPos[JointType_Count];
	cv::Point2d jointPoints[JointType_Count];
	UINT64 BodyID;
}BodyInfo;

//Store sensor out Body information
typedef struct SkeletonInfo{
	int Kinect_ID;
	int Count;												//현재 추적하고 있는 스켈레톤 갯수
	SYSTEMTIME	st;
	BodyInfo InfoBody[BODY_COUNT];
}SkeletonInfo;

class Kinect2
{
public:
	Kinect2(void);
	~Kinect2(void);

	//Kinect Initialize
	HRESULT initializeKinect2();

	//Get Color Opencv Image. (1920*1080)
	//Image must allocated.(size: 1920*1080, CV_8UC4)
	cv::Mat GetColorImage();

	//Get Depth Opencv Image. (512*424)
	//Image must allocated. (size: 512*424, CV_8UC4)
	cv::Mat GetDepthImage();

	//Get Body index Image
	//Image must allocated. (size : 512*424, CV_8UC4)
	//return Depth Image - Body segmentation.
	cv::Mat GetBodyIndexImage();

	//Get Body joint position. (joint count : 25)
	//and Draw Opencv Image.
	//if mode = 1, Draw DepthScale. so src = DepthImage
	//else if mode = 2, Draw ColorScale. so src = ColorImage.
	void CreateSkeleton(SkeletonInfo *m_SkeletonInfo, Mat *src, int mode);

	//Return Kinect Unique ID - (not work yet. 2014.11.18)
	void GetKinectID(WCHAR *KinectID);

	cv::Mat calculateMappedFrame(int mode); //mode 0: color to depth mode 1: depth to color

	int GroundNormalDetection(void);
	void GroundPramInit(void);
	void GroundNormalDetection_finish(void);
	cv::Mat GetGridMap_UsingPlaneFiiting(MobileRobot *Pioneer);

	// Image frame Mat
	Mat colorRAWFrameMat;
	Mat depthRAWFrameMat;
	Mat BodyRAWFrameMat;
	Mat colorMappedFrameMat;

	Mat Map_t;

	//LocalGrid
	void getLocalGrid(cv::Mat Point_Grid, LocalGrid *Gridinfo, MobileRobot *Pioneer);
	void Add_Grid_Points(std::vector<pair<int, int>> *observed_points, cv::Mat *Point_Grid);
	cv::Mat drawPoints(std::map<int, int> *observed_points);
	cv::Mat drawPoints(cv::Mat *points, cv::Mat *RT_mat);
	cv::Mat drawPoints(cv::Mat *points, cv::Mat *RT_mat, cv::Mat *output_img);
	int test_grid_Db_num;
	double get_gapTH(double from, double to);
	double get_Rotation_from_mat(cv::Mat *RTmat);
	double initial_rotation_matcher(cv::Mat *from, cv::Mat *to, LocalGrid *Gridinfo, double *initial_gauss);

private:
	//Device
	IKinectSensor*			 m_pKinectSensor;
	ICoordinateMapper*		 m_pCoordinateMapper;

	//Frame reader
	IColorFrameReader*		 m_pColorFrameReader;
	IDepthFrameReader*		 m_pDepthFrameReader;
	IBodyFrameReader*		 m_pBodyFrameReader;
	IBodyIndexFrameReader*	 m_pBodyIndexFrameReader;

	//Coordinate
	ColorSpacePoint*         pColorCoodinate;
	DepthSpacePoint*         pDepthCoordinate;

	// Frame data buffers
	RGBQUAD* pColorRAWBuffer;
	ushort* pDepthRAWBuffer;
	BYTE* pBodyRAWBuffer;

	/*RGBQUAD*				 m_pColorRGBX;
	RGBQUAD*				 m_pDepthRGBX;
	RGBQUAD*                 m_pOutputRGBX;
	RGBQUAD*                 m_pBackgroundRGBX; */

	int MapDepthToByte;
	int SkeletonCount;
	WCHAR UniqueID[256];

	void ConvertOpencvGrayImage(cv::Mat *src, UINT16* pBuffer, int nHeight, int nWidth, int nDepthMinReliableDistance, int nDepthMaxDistance);
	void ProcessSkel(SkeletonInfo* m_SkeletonInfo, int nBodyCount, IBody** ppBodies, Mat *src, int mode);

	void DrawSkelToMat(Mat *src, Point2d *JointPoints, Joint* pJoints, int mode, int t_id);
	void DrawSkelBone(Mat *src, Joint* pJoints, Point2d* pJointPoints, JointType joint0, JointType joint1, Scalar t_Color);

	//Draw Hand state. - but not implemented.
	void DrawHand(Mat *src, HandState handState, Point2d& handposition);

	//Change CameraSpace coordinate to DepthCoordinate / ColorCoorinate.
	Point2d BodyToScreen(const CameraSpacePoint& bodyPoint, int mode);

	// Release function
	template< class T > void SafeRelease(T** ppT);

	//Ground
	double *meanval_a, *meanval_b, *meanval_c, *normal1, *normal2, *normal3;
	int *Ground_th, *Ground_x, *Ground_y;
	cv::Mat *Rot_fit;
	int *Ground_flag;
	Mat G_point;
	Mat G_point_Rot;

	//Local Grid Points DB
	std::vector<Grid_Points_info> Localmap_PointDB;

	//Number_of_sensor
	int SensorCount;
};
Kinect2::Kinect2(void)
{
	m_pKinectSensor = nullptr;
	m_pCoordinateMapper = nullptr;
	m_pColorFrameReader = nullptr;
	m_pDepthFrameReader = nullptr;
	m_pBodyFrameReader = nullptr;
	m_pBodyIndexFrameReader = nullptr;

	//Allocate buffers
	pColorRAWBuffer = new RGBQUAD[KINECT_COLOR_HEIGHT * KINECT_COLOR_WIDTH];
	pDepthRAWBuffer = new ushort[KINECT_DEPTH_HEIGHT * KINECT_DEPTH_WIDTH];
	pBodyRAWBuffer = new BYTE[KINECT_DEPTH_HEIGHT * KINECT_DEPTH_WIDTH];
	pColorCoodinate = new ColorSpacePoint[KINECT_DEPTH_HEIGHT * KINECT_DEPTH_WIDTH];
	pDepthCoordinate = new DepthSpacePoint[KINECT_COLOR_HEIGHT * KINECT_COLOR_WIDTH];

	// Set 0
	memset(pColorRAWBuffer, 0, KINECT_COLOR_HEIGHT * KINECT_COLOR_WIDTH * sizeof(RGBQUAD));
	memset(pDepthRAWBuffer, 0, KINECT_DEPTH_HEIGHT * KINECT_DEPTH_WIDTH * sizeof(ushort));
	memset(pBodyRAWBuffer, 0, KINECT_DEPTH_HEIGHT * KINECT_DEPTH_WIDTH * sizeof(BYTE));

	//Set Mat
	colorRAWFrameMat = Mat(Size(KINECT_COLOR_WIDTH, KINECT_COLOR_HEIGHT), CV_8UC4, (void*)pColorRAWBuffer);
	depthRAWFrameMat = Mat(Size(KINECT_DEPTH_WIDTH, KINECT_DEPTH_HEIGHT), CV_16UC1, (void*)pDepthRAWBuffer);
	BodyRAWFrameMat = Mat(Size(KINECT_DEPTH_WIDTH, KINECT_DEPTH_HEIGHT), CV_8UC1, (void*)pBodyRAWBuffer);

	Map_t = Mat(Size(KINECT_DEPTH_WIDTH, KINECT_DEPTH_HEIGHT), CV_8UC4);

	//Ground param
	SensorCount = 1;
	meanval_a = new double[SensorCount];
	meanval_b = new double[SensorCount];
	meanval_c = new double[SensorCount];
	normal1 = new double[SensorCount];
	normal2 = new double[SensorCount];
	normal3 = new double[SensorCount];
	Ground_th = new int[SensorCount];
	Ground_x = new int[SensorCount];
	Ground_y = new int[SensorCount];
	Ground_flag = new int[SensorCount];
	memset(meanval_a, 0, sizeof(double)*SensorCount);
	memset(meanval_b, 0, sizeof(double)*SensorCount);
	memset(meanval_c, 0, sizeof(double)*SensorCount);
	memset(normal1, 0, sizeof(double)*SensorCount);
	memset(normal2, 0, sizeof(double)*SensorCount);
	memset(normal3, 0, sizeof(double)*SensorCount);
	memset(Ground_th, 0, sizeof(int)*SensorCount);
	memset(Ground_x, 0, sizeof(int)*SensorCount);
	memset(Ground_y, 0, sizeof(int)*SensorCount);
	memset(Ground_flag, 0, sizeof(int)*SensorCount);
	Rot_fit = new cv::Mat[SensorCount];

	test_grid_Db_num = 0;

	MapDepthToByte = 8000 / 256;
}
Kinect2::~Kinect2(void)
{
	// Release buffers
	if (pColorRAWBuffer)
	{
		delete pColorRAWBuffer;
		pColorRAWBuffer = nullptr;
	}
	if (pDepthRAWBuffer)
	{
		delete pDepthRAWBuffer;
		pDepthRAWBuffer = nullptr;
	}
	if (pBodyRAWBuffer)
	{
		delete pBodyRAWBuffer;
		pBodyRAWBuffer = nullptr;
	}
	if (pColorCoodinate)
	{
		delete pColorCoodinate;
		pColorCoodinate = nullptr;
	}
	if (pDepthCoordinate)
	{
		delete pDepthCoordinate;
		pDepthCoordinate = nullptr;
	}

	if (m_pKinectSensor){
		m_pKinectSensor->Close();
		SafeRelease(&m_pKinectSensor);
	}

	delete meanval_a;
	delete meanval_b;
	delete meanval_c;
	delete normal1;
	delete normal2;
	delete normal3;
	delete Ground_th;
	delete Ground_x;
	delete Ground_y;
	delete Ground_flag;
	delete Rot_fit;
}
template< class T > void Kinect2::SafeRelease(T** ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT = nullptr;
	}
}
HRESULT Kinect2::initializeKinect2(){
	printf("Start Kinect2 Initialize...\n");
	HRESULT hrc = GetDefaultKinectSensor(&m_pKinectSensor);				//Kinect Color Frame
	HRESULT hrd = NULL;													//Kinect Depth Frame
	HRESULT hrs = NULL;

	if (FAILED(hrc)){
		printf("Kinect2 Seonsor Open fail. Check Kinect Sensor connected.\n");
		return hrc;
	}

	if (m_pKinectSensor){
		// Initialize the Kinect and get the color reader
		IColorFrameSource*	pColorFrameSource = NULL;
		IDepthFrameSource*	pDepthFrameSource = NULL;
		IBodyFrameSource*	pBodyFrameSource = NULL;
		IBodyIndexFrameSource*	pBodyIndexFrameSource = NULL;

		hrc = m_pKinectSensor->Open();
		printf("Kinect2 Sensor open Complete!\n");

		if (SUCCEEDED(hrc)){
			hrc = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
			hrd = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);

			m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
			hrs = m_pKinectSensor->get_BodyFrameSource(&pBodyFrameSource);
			m_pKinectSensor->get_BodyIndexFrameSource(&pBodyIndexFrameSource);				//body index Frame source open
		}

		if (SUCCEEDED(hrc) && SUCCEEDED(hrd) && SUCCEEDED(hrs)){
			hrc = pColorFrameSource->OpenReader(&m_pColorFrameReader);
			hrd = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
			hrs = pBodyFrameSource->OpenReader(&m_pBodyFrameReader);
			pBodyIndexFrameSource->OpenReader(&m_pBodyIndexFrameReader);					//body index Frame reader open
		}

		if (SUCCEEDED(hrc)){
			//WCHAR temp[256] = L"";
			hrc = m_pKinectSensor->get_UniqueKinectId(_countof(UniqueID), UniqueID);
			printf("Kinect ID : %s\n", UniqueID);
		}

	}

	if (!m_pKinectSensor || FAILED(hrc) || FAILED(hrd) || FAILED(hrs)){
		printf("No ready Kinect2 found!\n");
		return E_FAIL;
	}

	//Ground Param Load
	GroundPramInit();

	printf("Kinect initialize Complete\n");

	return hrc;


	//printf("Start Kinect2 Initialize...\n");

	//HRESULT hr;

	//hr = GetDefaultKinectSensor(&m_pKinectSensor);

	//if(FAILED(hr)){
	//	printf("Kinect2 Seonsor Open fail. Check Kinect2 Sensor connected.\n");
	//	return hr;
	//}

	//if (m_pKinectSensor)
	//{
	//	// Initialize the Kinect and get coordinate mapper and the frame reader

	//	if (SUCCEEDED(hr))
	//	{
	//		hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
	//	}

	//	hr = m_pKinectSensor->Open();

	//	if (SUCCEEDED(hr))
	//	{
	//		hr = m_pKinectSensor->OpenMultiSourceFrameReader(
	//			FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color | FrameSourceTypes::FrameSourceTypes_Body,
	//			&m_pMultiSourceFrameReader);
	//	}
	//}

	//if (!m_pKinectSensor || FAILED(hr))
	//{
	//	printf("No ready Kinect2 found!\n");
	//	return E_FAIL;
	//}

	//printf("Kinect initialize Complete\n");

	//return hr;
}
cv::Mat Kinect2::GetColorImage(){

	cv::Mat RGB_img = cv::Mat::zeros(KINECT_COLOR_HEIGHT, KINECT_COLOR_WIDTH, CV_8UC4);

	if (!m_pColorFrameReader){
		printf("Kinect2: Create RGB Image Failed\n");
		return RGB_img;
	}

	IColorFrame* pColorFrame = nullptr;
	HRESULT hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);

	if (SUCCEEDED(hr)){
		INT64 nTime = 0;
		IFrameDescription* pColorFrameDescription = nullptr;
		int colorFrameWidth = 0;
		int colorFrameHeight = 0;
		ColorImageFormat ImageFormat = ColorImageFormat_None;
		UINT colorBufferSize = 0;
		RGBQUAD *pTmpColorBuffer = nullptr;

		hr = pColorFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr)){
			hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
		}

		//Check Color Image Width & Height
		if (SUCCEEDED(hr)){
			pColorFrameDescription->get_Width(&colorFrameWidth);
			pColorFrameDescription->get_Height(&colorFrameHeight);
			hr = pColorFrame->get_RawColorImageFormat(&ImageFormat);
		}

		if (SUCCEEDED(hr)){
			if (ImageFormat == ColorImageFormat_Bgra){
				//if Image format is BGRA -> copy image direct.
				hr = pColorFrame->AccessRawUnderlyingBuffer(&colorBufferSize, reinterpret_cast<BYTE**>(&pTmpColorBuffer));
			}
			else if (pColorRAWBuffer){
				//Default Image format Yuy2
				pTmpColorBuffer = pColorRAWBuffer;
				colorBufferSize = KINECT_COLOR_HEIGHT * KINECT_COLOR_WIDTH * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(colorBufferSize, reinterpret_cast<BYTE*>(pTmpColorBuffer), ColorImageFormat_Bgra);
			}
			else
				hr = E_FAIL;
		}

		//Image Format Convert
		if (SUCCEEDED(hr)){
			// Check color buffer size
			if (colorBufferSize == KINECT_COLOR_HEIGHT * KINECT_COLOR_WIDTH  * sizeof(RGBQUAD))
			{
				// Copy color data
				memcpy(pColorRAWBuffer, pTmpColorBuffer, colorBufferSize);
			}
		}
		flip(colorRAWFrameMat, RGB_img, 1);
		SafeRelease(&pColorFrameDescription);
	}

	SafeRelease(&pColorFrame);

	return RGB_img;
}
cv::Mat Kinect2::GetDepthImage(){

	cv::Mat Depth_img = cv::Mat::zeros(KINECT_DEPTH_HEIGHT, KINECT_DEPTH_WIDTH, CV_8UC1);
	cv::Mat Depth_img_tmp = cv::Mat::zeros(KINECT_DEPTH_HEIGHT, KINECT_DEPTH_WIDTH, CV_8UC1);

	if (!m_pDepthFrameReader){
		printf("Kinect2: Create Depth Image Failed\n");
		return Depth_img;
	}

	IDepthFrame* pDepthFrame = nullptr;

	HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);

	if (SUCCEEDED(hr)){
		INT64 nTime = 0;
		IFrameDescription* pDepthFrameDescription = nullptr;
		int depthFrameWidth = 0;
		int depthFrameHeight = 0;
		USHORT nDepthMinReliableDistance = 0;
		USHORT nDepthMaxDistance = 0;
		UINT depthBufferSize = 0;
		UINT16 *pTmpDepthBuffer = nullptr;

		hr = pDepthFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
		}

		if (SUCCEEDED(hr)){
			pDepthFrameDescription->get_Width(&depthFrameWidth);
			pDepthFrameDescription->get_Height(&depthFrameHeight);
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);

			// In order to see the full range of depth (including the less reliable far field depth)
			// we are setting nDepthMaxDistance to the extreme potential depth threshold
			nDepthMaxDistance = USHRT_MAX;

			//// 각 어플리케이션에서 최장거리 제한이 필요한 경우 아래 코드 수정..
			// Note:  If you wish to filter by reliable depth distance, uncomment the following line.
			//// hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
		}

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->AccessUnderlyingBuffer(&depthBufferSize, &pTmpDepthBuffer);
		}

		////////////////////////////////////////////////

		////////////////////////////////////////////////

		if (SUCCEEDED(hr)){
			// Check depth buffer size
			if (depthBufferSize == KINECT_DEPTH_WIDTH * KINECT_DEPTH_HEIGHT)
			{
				// Copy depth data
				memcpy(pDepthRAWBuffer, pTmpDepthBuffer, depthBufferSize * sizeof(ushort));
				ConvertOpencvGrayImage(&Depth_img, pTmpDepthBuffer, depthFrameHeight, depthFrameWidth, nDepthMinReliableDistance, nDepthMaxDistance);
			}

		}

		flip(Depth_img, Depth_img, 1);
		SafeRelease(&pDepthFrameDescription);
	}
	SafeRelease(&pDepthFrame);

	return Depth_img;
}
void Kinect2::ConvertOpencvGrayImage(cv::Mat *src, UINT16* pBuffer, int nHeight, int nWidth, int nMinDepth, int nMaxDepth){
	src->setTo(Scalar::all(0));
	if (pBuffer){
		//RGBQUAD* pRGBX = m_pDepthRGBX;

		const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);


		for (int i = 0; i < nWidth*nHeight; i++){
			USHORT depth = *pBuffer;

			src->data[i] = (byte)(depth >= nMinDepth && depth <= nMaxDepth ? (depth / MapDepthToByte) : 0);

			pBuffer++;
		}

	}
}
cv::Mat Kinect2::GetBodyIndexImage(){

	cv::Mat Body_img = cv::Mat::zeros(KINECT_DEPTH_HEIGHT, KINECT_DEPTH_WIDTH, CV_8UC1);

	if (!m_pBodyIndexFrameReader){
		printf("Kinect2: Create Body Image Failed\n");
		return Body_img;
	}

	IBodyIndexFrame* pBodyIndexFrame = nullptr;

	HRESULT hr = m_pBodyIndexFrameReader->AcquireLatestFrame(&pBodyIndexFrame);

	if (SUCCEEDED(hr)){
		INT64 nTime = 0;
		IFrameDescription* pBodyFrameDescription = nullptr;
		int bodyFrameWidth = 0;
		int bodyFrameHeight = 0;
		UINT bodyBufferSize = 0;
		BYTE *pTmpBodyBuffer = nullptr;

		hr = pBodyIndexFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr)){
			hr = pBodyIndexFrame->get_FrameDescription(&pBodyFrameDescription);
		}

		if (SUCCEEDED(hr)){
			pBodyFrameDescription->get_Width(&bodyFrameWidth);
			pBodyFrameDescription->get_Height(&bodyFrameHeight);
			hr = pBodyIndexFrame->AccessUnderlyingBuffer(&bodyBufferSize, &pTmpBodyBuffer);
		}

		if (SUCCEEDED(hr)){
			if (bodyBufferSize == KINECT_DEPTH_WIDTH * KINECT_DEPTH_HEIGHT)
			{
				memcpy(pBodyRAWBuffer, pTmpBodyBuffer, bodyBufferSize * sizeof(BYTE));
				BodyRAWFrameMat.copyTo(Body_img);
			}
		}

		SafeRelease(&pBodyFrameDescription);
	}

	SafeRelease(&pBodyIndexFrame);

	flip(Body_img, Body_img, 1);

	return Body_img;
}
void Kinect2::CreateSkeleton(SkeletonInfo* m_SkeletonInfo, Mat *src, int mode){
	m_SkeletonInfo->Count = -1;


	if (!m_pBodyFrameReader){
		printf("Kinect2: Create skeleton Failed\n");
		return;
	}

	IBodyFrame* pBodyFrame = NULL;

	HRESULT hr = m_pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);

	if (SUCCEEDED(hr)){
		INT64 nTime = 0;

		hr = pBodyFrame->get_RelativeTime(&nTime);

		IBody* ppBodies[BODY_COUNT] = { 0 };

		if (SUCCEEDED(hr)){
			hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);
		}

		//Process raw skeleton data
		if (SUCCEEDED(hr)){
			ProcessSkel(m_SkeletonInfo, BODY_COUNT, ppBodies, src, mode);												//process & draw body
		}

		for (int i = 0; i < _countof(ppBodies); i++){
			SafeRelease(&ppBodies[i]);
		}
	}


	SafeRelease(&pBodyFrame);
}
void Kinect2::DrawSkelToMat(cv::Mat *src, Point2d* JointPoints, Joint* pJoint, int mode, int t_id){
	Scalar t_Color = Scalar((t_id * 37) % 256, (t_id * 113) % 256, (t_id * 71) % 256);

	// Torso
	DrawSkelBone(src, pJoint, JointPoints, JointType_Head, JointType_Neck, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_Neck, JointType_SpineShoulder, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_SpineShoulder, JointType_SpineMid, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_SpineMid, JointType_SpineBase, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_SpineShoulder, JointType_ShoulderRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_SpineShoulder, JointType_ShoulderLeft, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_SpineBase, JointType_HipRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_SpineBase, JointType_HipLeft, t_Color);

	// Right Arm
	DrawSkelBone(src, pJoint, JointPoints, JointType_ShoulderRight, JointType_ElbowRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_ElbowRight, JointType_WristRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_WristRight, JointType_HandRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_HandRight, JointType_HandTipRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_WristRight, JointType_ThumbRight, t_Color);

	// Left Arm
	DrawSkelBone(src, pJoint, JointPoints, JointType_ShoulderLeft, JointType_ElbowLeft, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_ElbowLeft, JointType_WristLeft, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_WristLeft, JointType_HandLeft, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_HandLeft, JointType_HandTipLeft, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_WristLeft, JointType_ThumbLeft, t_Color);

	// Right Leg
	DrawSkelBone(src, pJoint, JointPoints, JointType_HipRight, JointType_KneeRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_KneeRight, JointType_AnkleRight, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_AnkleRight, JointType_FootRight, t_Color);

	// Left Leg
	DrawSkelBone(src, pJoint, JointPoints, JointType_HipLeft, JointType_KneeLeft, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_KneeLeft, JointType_AnkleLeft, t_Color);
	DrawSkelBone(src, pJoint, JointPoints, JointType_AnkleLeft, JointType_FootLeft, t_Color);

	//Draw the Joints	- not implemented.
}
void Kinect2::DrawSkelBone(Mat *src, Joint* pJoints, Point2d* pJointPoints, JointType joint0, JointType joint1, Scalar t_Color){

	static const float c_JointThickness = 3.0f;
	static const float c_TrackedBoneThickness = 6.0f;
	static const float c_InferredBoneThickness = 1.0f;
	static const float c_HandSize = 30.f;

	TrackingState joint0State = pJoints[joint0].TrackingState;
	TrackingState joint1State = pJoints[joint1].TrackingState;

	// If we can't find either of these joints, exit
	if ((joint0State == TrackingState_NotTracked) || (joint1State == TrackingState_NotTracked))
	{
		return;
	}

	// Don't draw if both points are inferred
	if ((joint0State == TrackingState_Inferred) && (joint1State == TrackingState_Inferred))
	{
		return;
	}

	// We assume all drawn bones are inferred unless BOTH joints are tracked
	if ((joint0State == TrackingState_Tracked) && (joint1State == TrackingState_Tracked))
	{
		line(*src, pJointPoints[joint0], pJointPoints[joint1], t_Color, c_TrackedBoneThickness);
	}
	else
	{
		line(*src, pJointPoints[joint0], pJointPoints[joint1], t_Color, c_InferredBoneThickness);
	}
}

//Not Implemented. - not work in SDK 2.0 preview version.. (나중에 사용해야할듯 handState가 항상 unknown.)
void Kinect2::DrawHand(Mat *src, HandState handState, Point2d& handposition){
	switch (handState){
	case HandState_Closed:
		break;
	case HandState_Open:
		break;
	case HandState_Lasso:
		break;
	}
}
Point2d Kinect2::BodyToScreen(const CameraSpacePoint& bodyPoint, int mode){
	Point2d Return_val;

	if (mode == 0){
		ColorSpacePoint colorPoint = { 0 };
		m_pCoordinateMapper->MapCameraPointToColorSpace(bodyPoint, &colorPoint);

		Return_val.x = static_cast<double>(KINECT_COLOR_WIDTH - colorPoint.X);
		Return_val.y = static_cast<double>(colorPoint.Y);
	}
	if (mode == 1){
		DepthSpacePoint depthPoint = { 0 };
		m_pCoordinateMapper->MapCameraPointToDepthSpace(bodyPoint, &depthPoint);

		Return_val.x = static_cast<double>(KINECT_DEPTH_WIDTH - depthPoint.X);
		Return_val.y = static_cast<double>(depthPoint.Y);
	}

	return Return_val;
}
void Kinect2::ProcessSkel(SkeletonInfo* m_SkeletonInfo, int nBodyCount, IBody** ppBodies, Mat *src, int mode){
	SkeletonCount = 0;
	HRESULT hr;

	m_SkeletonInfo->Count = SkeletonCount;

	if (m_pCoordinateMapper){
		for (int i = 0; i < nBodyCount; i++){
			IBody* pBody = ppBodies[i];

			if (pBody){
				BOOLEAN bTracked = false;
				hr = pBody->get_IsTracked(&bTracked);

				if (SUCCEEDED(hr) && bTracked){
					Joint joints[JointType_Count];
					Point2d jointPoints[JointType_Count];
					HandState leftHandState = HandState_Unknown;
					HandState rightHandState = HandState_Unknown;

					pBody->get_HandLeftState(&leftHandState);
					pBody->get_HandRightState(&rightHandState);

					hr = pBody->GetJoints(_countof(joints), joints);

					//Get Tracking Body ID;
					UINT64 t_ID;
					hr = pBody->get_TrackingId(&t_ID);
					m_SkeletonInfo->InfoBody[SkeletonCount].BodyID = t_ID;

					//cout<<"body_idx : "<<i<<" ";
					if (SUCCEEDED(hr)){

						for (int j = 0; j < _countof(joints); j++){
							m_SkeletonInfo->InfoBody[SkeletonCount].JointPos[j] = joints[j];
							jointPoints[j] = BodyToScreen(joints[j].Position, mode);
							//cout<<jointPoints[j]<<" ";
						}
						//cout<<endl;

						//printf("%f %f\n", jointPoints[0]);

						DrawSkelToMat(src, jointPoints, joints, mode, t_ID);

						DrawHand(src, leftHandState, jointPoints[JointType_HandLeft]);
						DrawHand(src, rightHandState, jointPoints[JointType_HandRight]);

						SkeletonCount++;
					}
				}
			}
		}

		m_SkeletonInfo->Count = SkeletonCount;
	}
}
void Kinect2::GetKinectID(WCHAR *KinectID){
	memcpy(KinectID, UniqueID, _countof(UniqueID)*sizeof(WCHAR));
}
cv::Mat Kinect2::calculateMappedFrame(int mode)
{
	cv::Mat Mapping_img = cv::Mat::zeros(KINECT_DEPTH_HEIGHT, KINECT_DEPTH_WIDTH, CV_8UC4);
	HRESULT hr = E_FAIL;

	// Depth coordinate mapping
	if (m_pCoordinateMapper != nullptr)
	{

		// Color 2 Depth Mapping
		if (mode == 0)
		{
			hr = m_pCoordinateMapper->MapDepthFrameToColorSpace(KINECT_DEPTH_WIDTH * KINECT_DEPTH_HEIGHT, (UINT16*)pDepthRAWBuffer, KINECT_DEPTH_WIDTH * KINECT_DEPTH_HEIGHT, pColorCoodinate);
			if (SUCCEEDED(hr))
			{
				Vec4b* p = Map_t.ptr< Vec4b >(0);
				for (int idx = 0; idx < KINECT_DEPTH_WIDTH * KINECT_DEPTH_HEIGHT; idx++)
				{
					ColorSpacePoint csp = pColorCoodinate[idx];
					int colorX = (int)floor(csp.X + 0.5);
					int colorY = (int)floor(csp.Y + 0.5);
					if (colorX >= 0 && colorX < KINECT_COLOR_WIDTH && colorY >= 0 && colorY < KINECT_COLOR_HEIGHT)
					{
						p[idx] = colorRAWFrameMat.at< Vec4b >(colorY, colorX);
					}
				}
				Map_t.copyTo(Mapping_img);
				flip(Mapping_img, Mapping_img, 1);
			}
		}
		// Depth 2 Color Mapping
		else if (mode == 1)
		{
			hr = m_pCoordinateMapper->MapColorFrameToDepthSpace(KINECT_DEPTH_WIDTH * KINECT_DEPTH_HEIGHT, (UINT16*)pDepthRAWBuffer, KINECT_COLOR_WIDTH * KINECT_COLOR_HEIGHT, pDepthCoordinate);
			if (SUCCEEDED(hr))
			{
				colorMappedFrameMat = Mat::zeros(Size(KINECT_COLOR_WIDTH, KINECT_COLOR_HEIGHT), CV_8UC4);
				Vec4b* pMappedFrame = colorMappedFrameMat.ptr< Vec4b >(0);
#pragma omp parallel for

				for (int idx = 0; idx < KINECT_COLOR_WIDTH * KINECT_COLOR_HEIGHT; idx++)
				{
					DepthSpacePoint dsp = pDepthCoordinate[idx];
					int depthX = (int)floor(dsp.X + 0.5);
					int depthY = (int)floor(dsp.Y + 0.5);
					if (depthX >= 0 && depthX < KINECT_DEPTH_WIDTH && depthY >= 0 && depthY < KINECT_DEPTH_HEIGHT)
					{
						ushort val = pDepthRAWBuffer[depthY * KINECT_DEPTH_WIDTH + depthX];
						if (val)
							pMappedFrame[idx] = ((Vec4b*)pColorRAWBuffer)[idx];
						else
							pMappedFrame[idx] = Vec4b(0, 0, 0, 0);
					}
				}
				colorMappedFrameMat.copyTo(Map_t);
				flip(Mapping_img, Mapping_img, 1);
			}
		}
	}

	return Mapping_img;
}
int Kinect2::GroundNormalDetection(void)
{
	//int return_flag = 0;
	//return 0;
	int sensorNumber = 0; //for mutiple Kinect2

	if (Ground_flag[sensorNumber] == 1)
		return 1;

	cv::Mat Depth_img = cv::Mat::zeros(KINECT_DEPTH_HEIGHT, KINECT_DEPTH_WIDTH, CV_8UC4);
	cv::Mat Depth_img_tmp = cv::Mat::zeros(KINECT_DEPTH_HEIGHT, KINECT_DEPTH_WIDTH, CV_8UC1);

	if (!m_pDepthFrameReader)
	{
		printf("Kinect2: Create Depth Image Failed\n");
		return 0;
	}

	IDepthFrame* pDepthFrame = NULL;

	HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		USHORT nDepthMinReliableDistance = 0;
		USHORT nDepthMaxDistance = 0;
		UINT nBufferSize = 0;
		UINT16 *pBuffer = NULL;

		hr = pDepthFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr)){
			pFrameDescription->get_Width(&nWidth);
			pFrameDescription->get_Height(&nHeight);
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);

			// In order to see the full range of depth (including the less reliable far field depth)
			// we are setting nDepthMaxDistance to the extreme potential depth threshold
			nDepthMaxDistance = USHRT_MAX;

			//// 각 어플리케이션에서 최장거리 제한이 필요한 경우 아래 코드 수정..
			// Note:  If you wish to filter by reliable depth distance, uncomment the following line.
			//// hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
		}

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
		}

		if (SUCCEEDED(hr)){
			ConvertOpencvGrayImage(&Depth_img_tmp, pBuffer, nHeight, nWidth, nDepthMinReliableDistance, nDepthMaxDistance);
		}
		flip(Depth_img_tmp, Depth_img_tmp, 1);
		char windowName[64];
		sprintf(windowName, "Kinect2 Ground sensor 0");
		cv::imshow(windowName, Depth_img_tmp);
		cvSetMouseCallback(windowName, my_mouse_callback_ground);

		if (drawing_box_flag == -1)
		{
			drawing_box_flag = 0;

			int count_data=0;
			for (int y = Ground_box.y; y<Ground_box.y + Ground_box.height; y = y + 1)
			{
				for (int x = Ground_box.x; x<Ground_box.x + Ground_box.width; x++)
				{
					int index = y*KINECT_DEPTH_WIDTH + (KINECT_DEPTH_WIDTH - x - 1);
					float val = (float)pBuffer[index];
					//float inputdepth = (float)val;
					if (val > nDepthMinReliableDistance)
					{
						count_data++;
					}
				}
			}

			const int mySizes[3] = { count_data, 1, 3 };
			cv::Mat_<double> f(count_data, 3);

			for (int y = Ground_box.y, i=0; y<Ground_box.y + Ground_box.height; y = y + 1)
			{
				for (int x = Ground_box.x; x<Ground_box.x + Ground_box.width; x = x + 1)
				{

					int index = y*KINECT_DEPTH_WIDTH + (KINECT_DEPTH_WIDTH - x - 1);
					float inputdepth = (float)pBuffer[index];;
					if (inputdepth > nDepthMinReliableDistance)
					{
						int index = y*KINECT_DEPTH_WIDTH + (KINECT_DEPTH_WIDTH - x - 1);
						DepthSpacePoint depthspacePoint = { (KINECT_DEPTH_WIDTH - x - 1), y };
						CameraSpacePoint outputPoint;

						m_pCoordinateMapper->MapDepthPointToCameraSpace(depthspacePoint, pBuffer[index], &outputPoint);

						f.row(i)(0) = 1000.0f*outputPoint.X;
						f.row(i)(1) = 1000.0f*outputPoint.Y;
						f.row(i)(2) = 1000.0f*outputPoint.Z;
						i++;
					}
				}
			}
			cv::Mat_<double> mean;
			cv::PCA pca(f, mean, CV_PCA_DATA_AS_ROW);
			double p_to_plane_thresh = pca.eigenvalues.at<double>(2);
			cv::Vec3d basis_x = pca.eigenvectors.row(0);
			cv::Vec3d basis_y = pca.eigenvectors.row(1);
			cv::Vec3d nrm = pca.eigenvectors.row(2);
			nrm = nrm / norm(nrm);
			cv::Vec3d x0 = pca.mean;

			int state;
			char Ground_file_name[128];
			sprintf(Ground_file_name, "..\\data\\Kinect2_Ground_pram00.txt");
			FILE * file = fopen(Ground_file_name, "wt");
			if (file == NULL)
			{
				cout << "file open error!\n" << endl;
			}
			fprintf(file, "%3.5f %3.5f %3.5f %3.5f %3.5f %3.5f", x0[0], x0[1], x0[2], nrm[0], nrm[1], nrm[2]);
			state = fclose(file);
			if (state != 0)
			{
				cout << "file close error!\n" << endl;
			}
			cvDestroyWindow(windowName);
			Ground_flag[sensorNumber] = 1;
		}
		SafeRelease(&pFrameDescription);
	}

	SafeRelease(&pDepthFrame);

	//flip(Depth_img_tmp, Depth_img_tmp, 1);
	//cvtColor(Depth_img_tmp, Depth_img, CV_BGRA2GRAY, 0);

	int return_flag = 1;
	for (int i = 0; i < SensorCount; i++)
		if (Ground_flag[i] == 0)
			return_flag = 0;

	return return_flag;
}
void Kinect2::GroundPramInit(void)
{
	ifstream MyFile;
	for (int i = 0; i < SensorCount; i++)
	{
		char file_name[64];
		sprintf(file_name, "..\\data\\Kinect2_Ground_pram%02d.txt", i);

		MyFile.open(file_name, ios::in);
		MyFile >> meanval_a[i] >> meanval_b[i] >> meanval_c[i] >> normal1[i] >> normal2[i] >> normal3[i];
		MyFile.close();

		Mat Grond_plane = (Mat_<float>(1, 3) << 0, 1, 0);

		Mat Grond_nomal_sensing = (Mat_<float>(1, 3) << normal1[i], normal2[i], normal3[i]);
		Mat RotationAxis = (Mat_<float>(1, 3) << 0, 0, 0);
		RotationAxis = Grond_nomal_sensing.cross(Grond_plane);
		RotationAxis = RotationAxis / norm(RotationAxis);
		float RotationAngle_Radian = acos(Grond_plane.dot(Grond_nomal_sensing) / (norm(Grond_nomal_sensing)*norm(Grond_plane)));
		Mat Identity_mat = Identity_mat.eye(3, 3, CV_32F);
		Mat p_p = RotationAxis.t() * RotationAxis;
		float a1 = RotationAxis.at<float>(0, 0);
		float a2 = RotationAxis.at<float>(0, 1);
		float a3 = RotationAxis.at<float>(0, 2);
		Mat P_P = (Mat_<float>(3, 3) << 0, a3, -a2, a3, 0, a1, a2, -a1, 0);
		Rot_fit[i] = cos(RotationAngle_Radian)*Identity_mat + (1 - cos(RotationAngle_Radian))*p_p - sin(RotationAngle_Radian)*P_P;
		//cout << "---------Kinect2 Grond_planeBySensor " << i << "-----------" << endl;
		//cout << "Grond_plane=" << Grond_plane << endl << endl;
		//cout << "Grond_nomal_sensing=" << Grond_nomal_sensing << endl << endl;
		//cout << "RotationAxis=" << RotationAxis << endl;
		//cout << "RotationAngle_Radian=" << RotationAngle_Radian << endl;
		//cout << "Identity_mat=" << Identity_mat << endl;
		//cout << "p_p=" << p_p << endl;
		//cout << "P_P=" << P_P << endl;
		//cout << "Rot_G_fit=" << Rot_fit[i] << endl;
		//cout << "---------Kinect2 Grond_planeBySensor " << i << "-----------" << endl;
	}

	G_point = (Mat_<float>(1, 3) << 0, 0, 0);
	G_point_Rot = (Mat_<float>(1, 3) << 0, 0, 0);

	MyFile.open("..\\data\\Kinect2_Ground_pram_cal.txt", ios::in);
	for (int i = 0; i < SensorCount; i++)
	{
		MyFile >> Ground_x[i] >> Ground_y[i] >> Ground_th[i];
	}
	MyFile.close();
}
void Kinect2::GroundNormalDetection_finish(void)
{
	memset(Ground_flag, 0, sizeof(int)*SensorCount);
	GroundPramInit();
}
cv::Mat Kinect2::GetGridMap_UsingPlaneFiiting(MobileRobot *Pioneer)
{
	LocalGrid outputGrid_info;

	int GridsizeY = 480;
	int GridsizeX = 300;

	int Robot_posi_on_grid[2] = { GridsizeX / 2, GridsizeY - ori_posi_of_robot };
	int Robot_size_on_gird[2] = { 15, 10 };
	outputGrid_info.robot_posi_in_cols = GridsizeX / 2;
	outputGrid_info.robot_posi_in_rows = GridsizeY - ori_posi_of_robot;

	ifstream MyFile;
	MyFile.open("..\\data\\Kinect2_Ground_pram_cal.txt", ios::in);
	for (int i = 0; i < SensorCount; i++)
		MyFile >> Ground_x[i] >> Ground_y[i] >> Ground_th[i];
	MyFile.close();

	double gridcellsize = GridsizeY / 10000.0;
	Mat Gridmap_free(GridsizeY, GridsizeX, CV_8UC3, cvScalar(125, 125, 125));
	Mat Gridmap_occupy(GridsizeY, GridsizeX, CV_8UC3, cvScalar(0, 0, 0));
	Mat Gridmap_occupy_plot(GridsizeY, GridsizeX, CV_8UC3, cvScalar(0, 0, 0));
	Mat Gridmap_occupy_people(GridsizeY, GridsizeX, CV_8UC3, cvScalar(0, 0, 0));
	Mat Gridmap2(GridsizeY, GridsizeX, CV_8UC3, cvScalar(0, 0, 0));
	outputGrid_info.Robot_Movement_rate = gridcellsize;
	int index;

	for (int Kinect_idx = 0; Kinect_idx<SensorCount; Kinect_idx++)
	{
		if (!m_pDepthFrameReader)
		{
			printf("Kinect2: Create Depth Image Failed\n");
			return Gridmap2;
		}

		IDepthFrame* pDepthFrame = NULL;

		HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);

		if (SUCCEEDED(hr))
		{
			INT64 nTime = 0;
			IFrameDescription* pFrameDescription = NULL;
			int nWidth = 0;
			int nHeight = 0;
			USHORT nDepthMinReliableDistance = 0;
			USHORT nDepthMaxDistance = 0;
			UINT nBufferSize = 0;
			UINT16 *pBuffer = NULL;

			hr = pDepthFrame->get_RelativeTime(&nTime);

			if (SUCCEEDED(hr)){
				hr = pDepthFrame->get_FrameDescription(&pFrameDescription);
			}

			if (SUCCEEDED(hr)){
				pFrameDescription->get_Width(&nWidth);
				pFrameDescription->get_Height(&nHeight);
				hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);

				// In order to see the full range of depth (including the less reliable far field depth)
				// we are setting nDepthMaxDistance to the extreme potential depth threshold
				nDepthMaxDistance = USHRT_MAX;

				//// 각 어플리케이션에서 최장거리 제한이 필요한 경우 아래 코드 수정..
				// Note:  If you wish to filter by reliable depth distance, uncomment the following line.
				//// hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
			}

			if (SUCCEEDED(hr)){
				hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
			}
						

			cv::Mat Point_Grid_map(GridsizeY, GridsizeX, CV_8UC1, cv::Scalar(0));
			int j = 0;
			int sampling_gap =3;
			for (int y = 0; y<nHeight; y = y + sampling_gap)
			{
				for (int x = nWidth; x >= 1; x = x - sampling_gap)
				{
					j = y*nWidth + ((nWidth - x));
					float val0 = (float)pBuffer[j];
					int test_int = (int)pBuffer[j];

					if (val0 >nDepthMinReliableDistance && val0<nDepthMaxDistance)
					{
						int index = y*KINECT_DEPTH_WIDTH + (KINECT_DEPTH_WIDTH - x );
						DepthSpacePoint depthspacePoint = { (/*KINECT_DEPTH_WIDTH - */x), y };
						CameraSpacePoint outputPoint;

						m_pCoordinateMapper->MapDepthPointToCameraSpace(depthspacePoint, pBuffer[index], &outputPoint);

						
						double x1 = 1000.0f*outputPoint.X - meanval_a[Kinect_idx];
						double x2 = 1000.0f*outputPoint.Y - meanval_b[Kinect_idx];
						double x3 = 1000.0f*outputPoint.Z - meanval_c[Kinect_idx];
						double error = abs(x1*normal1[Kinect_idx] + x2*normal2[Kinect_idx] + x3*normal3[Kinect_idx]);
						G_point.at<float>(0, 0) = 1000.0f*(float)outputPoint.X;
						G_point.at<float>(0, 1) = 1000.0f*(float)outputPoint.Y;
						G_point.at<float>(0, 2) = 1000.0f*(float)outputPoint.Z;

						G_point_Rot = Rot_fit[Kinect_idx] * G_point.t();
						x1 = G_point_Rot.at<float>(0, 0);
						x2 = G_point_Rot.at<float>(1, 0);
						x3 = G_point_Rot.at<float>(2, 0);


						int grid_Gx1 = (int)(ceil((x1 + (double)Ground_x[Kinect_idx]) * (gridcellsize)));
						int grid_Gx3 = GridsizeY - (int)(ceil((x3 + (double)Ground_y[Kinect_idx]) * (gridcellsize)));

						if (grid_Gx1 < GridsizeX && grid_Gx1 >= 0 && grid_Gx3 >= 0 && grid_Gx3 <GridsizeY)
						{
							if (error <  Ground_th[Kinect_idx])
							{
								cv::circle(Gridmap_free, cv::Point(grid_Gx1, grid_Gx3), 3, cv::Scalar(255, 255, 255), -5);
							}
							else
							{
								if (abs(grid_Gx1 - Robot_posi_on_grid[0])<Robot_size_on_gird[0] && abs(grid_Gx3 - Robot_posi_on_grid[1])<Robot_size_on_gird[1])
								{

								}
								else
								{
									Point_Grid_map.at<unsigned char>(grid_Gx3, grid_Gx1)=255;
									cv::circle(Gridmap_occupy, cv::Point(grid_Gx1, grid_Gx3), 1, cv::Scalar(255, 255, 255), -5);
								}
							}
						}
					}
				}
			}

			getLocalGrid(Point_Grid_map, &outputGrid_info, Pioneer);


			SafeRelease(&pFrameDescription);
		}
		SafeRelease(&pDepthFrame);
	}

	////cv::Mat output = Gridmap_free.clone();
	//output -= Gridmap_occupy;
	Gridmap2 = Gridmap_free - Gridmap_occupy;
	outputGrid_info.localGrid = Gridmap2.clone();
	outputGrid_info.localGrid_for_plot = Gridmap2.clone();
	cvtColor(Gridmap2, outputGrid_info.graylocalGrid, CV_RGB2GRAY);



	return Gridmap2;
}
void Kinect2::getLocalGrid(cv::Mat Point_Grid, LocalGrid *Gridinfo, MobileRobot *Pioneer)
{
	char Grid_Db_num[64];
	sprintf(Grid_Db_num, "..\\data\\GridMap\\%05d.jpg", test_grid_Db_num++);
	Point_Grid=cv::imread(Grid_Db_num);
	cvtColor(Point_Grid, Point_Grid, CV_RGB2GRAY);
	Point_Grid.convertTo(Point_Grid, CV_8UC1);

	int number_of_memorial = 50;

	std::vector<pair<int, int>> grid_points;
	Add_Grid_Points(&grid_points, &Point_Grid);

	Grid_Points_info observe;
	int sampling_gap = 1;
	observe.points_matrix = cv::Mat((int)grid_points.size() / sampling_gap, 2, CV_64FC1);
	observe.Point_Grid = Point_Grid.clone();
	int observe_Point_Counter = 0;
	for (std::vector<pair<int, int>>::iterator iter = grid_points.begin(); observe_Point_Counter<observe.points_matrix.rows; iter += sampling_gap, observe_Point_Counter++)
	{
		observe.points_matrix.at<double>(observe_Point_Counter, 0) = iter->first;
		observe.points_matrix.at<double>(observe_Point_Counter, 1) = iter->second;
	}
	observe.Robot_x = Pioneer->robot.getX();
	observe.Robot_y = Pioneer->robot.getY();
	observe.Robot_th = Pioneer->robot.getTh();
	observe.x = 0;
	observe.y = 0;
	observe.th = 0;
	observe.RT_Matrix = cv::Mat(2, 3, CV_64FC1, cv::Scalar(0));
	observe.RT_Matrix.at<double>(0, 0) = 1.0;
	observe.RT_Matrix.at<double>(1, 1) = 1.0;

	//cv::Mat observed_img_before = drawPoints(&observe.points_matrix, &observe.RT_Matrix);
	//cv::imshow("observed_img_before", observed_img_before);

	if ((int)grid_points.size() > 10)
	{
		if ((int)Localmap_PointDB.size() == 0)
		{
			Localmap_PointDB.push_back(observe);
		}
		else
		{
			double initial_gauss[3] = {};
			int Local_grid_index_befor = Localmap_PointDB.size() - 1;		

			double robot_Th_befor = -CV_PI*Localmap_PointDB.at(Local_grid_index_befor).Robot_th / 180.0;
			double x_gap = observe.Robot_x - Localmap_PointDB.at(Local_grid_index_befor).Robot_x;
			double y_gap = observe.Robot_y - Localmap_PointDB.at(Local_grid_index_befor).Robot_y;

			double RobotMovementRate = Gridinfo->Robot_Movement_rate;
			initial_gauss[0] = RobotMovementRate*(x_gap*cos(robot_Th_befor) - y_gap*sin(robot_Th_befor));
			initial_gauss[1] = RobotMovementRate*(x_gap*sin(robot_Th_befor) + y_gap*cos(robot_Th_befor));
			initial_gauss[2] = get_gapTH(Localmap_PointDB.at(Local_grid_index_befor).Robot_th, observe.Robot_th);

			initial_rotation_matcher(&Localmap_PointDB.at(Local_grid_index_befor).Point_Grid, &observe.Point_Grid, Gridinfo, initial_gauss);

			int num_observed_points = observe.points_matrix.rows;
			double *observed_Data = (double *)calloc(2 * num_observed_points, sizeof(double));
			std::memcpy(observed_Data, observe.points_matrix.data, sizeof(double) * 2 * num_observed_points);
			IcpPointToPlane icp(observed_Data, num_observed_points, 2);

			cv::Mat RT_matrix = cv::getRotationMatrix2D(cv::Point2f(Gridinfo->robot_posi_in_cols, Gridinfo->robot_posi_in_rows), initial_gauss[2], 1);
			double center_tran_x = RT_matrix.at<double>(0, 2);
			double center_tran_y = RT_matrix.at<double>(1, 2);
			RT_matrix.at<double>(0, 2) += initial_gauss[0];
			RT_matrix.at<double>(1, 2) += initial_gauss[1];
			ICP::Matrix R(2, 2);
			ICP::Matrix T(2, 1);
			R.setVal(RT_matrix.at<double>(0, 0), 0, 0, 0, 0);
			R.setVal(RT_matrix.at<double>(1, 0), 1, 0, 1, 0);
			R.setVal(RT_matrix.at<double>(0, 1), 0, 1, 0, 1);
			R.setVal(RT_matrix.at<double>(1, 1), 1, 1, 1, 1);
			T.setVal(RT_matrix.at<double>(0, 2), 0, 0, 0, 0);
			T.setVal(RT_matrix.at<double>(1, 2), 1, 0, 1, 0);

			int num_predicted_points = (int)Localmap_PointDB.at(Local_grid_index_befor).points_matrix.rows;
			double *predicted_Data = (double *)calloc(2 * num_predicted_points, sizeof(double));
			std::memcpy(predicted_Data, Localmap_PointDB.at(Local_grid_index_befor).points_matrix.data, sizeof(double) * 2 * num_predicted_points);
			icp.fit(predicted_Data, num_predicted_points, R, T,-1);

			cv::Mat result_RT_mat(2, 3, CV_64FC1, cv::Scalar(0));

			double temp_R_result[4];
			R.getData(temp_R_result, 0, 0);
			result_RT_mat.at<double>(0, 0) = temp_R_result[0];
			result_RT_mat.at<double>(0, 1) = temp_R_result[1];
			result_RT_mat.at<double>(1, 0) = temp_R_result[2];
			result_RT_mat.at<double>(1, 1) = temp_R_result[3];
			double rotation_val = get_Rotation_from_mat(&result_RT_mat);

			cv::Mat RT_result_matrix = cv::getRotationMatrix2D(cv::Point2f(Gridinfo->robot_posi_in_cols, Gridinfo->robot_posi_in_rows), -rotation_val, 1);

			double temp_T_result[2];
			T.getData(temp_T_result, 0, 0);
			result_RT_mat.at<double>(0, 2) = temp_T_result[0];
			result_RT_mat.at<double>(1, 2) = temp_T_result[1];
			//result_RT_mat.at<double>(0, 2) = RT_result_matrix.at<double>(0, 2);
			//result_RT_mat.at<double>(1, 2) = RT_result_matrix.at<double>(1, 2);

			//cv::Mat predicted_img = drawPoints(&Localmap_PointDB.at(Local_grid_index_befor).points_matrix, &observe.RT_Matrix);
			//cv::imshow("predicted_img", predicted_img);

			//cout << result_RT_mat << endl;
			//cout << RT_result_matrix << endl;

			result_RT_mat = RT_result_matrix.clone();

			observe.RT_Matrix = result_RT_mat.clone();
			observe.th = rotation_val;
			observe.x = result_RT_mat.at<double>(0, 2) - center_tran_x;
			observe.y = result_RT_mat.at<double>(1, 2) - center_tran_y;

			

			//printf("x:%7.2lf y:%7.2lf th:%7.2lf			", observe.x, observe.y, observe.th);

			Localmap_PointDB.push_back(observe);

			//cv::Mat predicted_img_c = drawPoints(&Localmap_PointDB.at(Local_grid_index_befor).points_matrix, &observe.RT_Matrix);
			//cv::imshow("predicted_img_corredted", predicted_img_c);

			//std::vector<cv::Mat> merged_img_array;
			//cv::Mat merged_img;
			//merged_img_array.push_back(predicted_img_c);
			//merged_img_array.push_back(predicted_img);
			//merged_img_array.push_back(observed_img_before);
			//cv::merge(merged_img_array, merged_img);
			//cv::resize(merged_img, merged_img, cv::Size(merged_img.cols * 2, merged_img.rows * 2));
			//cv::imshow("merged_img", merged_img);

			//cv::waitKey(0);
		}


		if (Localmap_PointDB.size() > number_of_memorial)
		{
			std::vector<Grid_Points_info>::iterator eraser = Localmap_PointDB.begin();
			Localmap_PointDB.erase(eraser);
		}
	}

	cv::Mat GridMap(480, 300, CV_8UC1, cv::Scalar(125));
	cv::Mat RT_mat = cv::Mat::eye(cv::Size(3, 3), CV_64FC1);
	for (int map_i = (int)Localmap_PointDB.size()-1; map_i >=0; map_i--)
	{
		cv::Mat img_warp_mat(2, 3, CV_64FC1);
		cv::Mat warping_point(3, 1, CV_64FC1);
		std::memcpy(img_warp_mat.data, RT_mat.data, sizeof(double) * 2 * 3);

		drawPoints(&Localmap_PointDB.at(map_i).points_matrix, &img_warp_mat, &GridMap);
	
		cv::Mat temp_RT_mat(3, 3, CV_64FC1, cv::Scalar(0));
		std::memcpy(temp_RT_mat.data, Localmap_PointDB.at(map_i).RT_Matrix.data, sizeof(double) * 2 * 3);
		temp_RT_mat.at<double>(2, 2) = 1.0;
		RT_mat = RT_mat*temp_RT_mat;
	
	
	}
	cv::imshow("test_Grid_map", GridMap);
	//cv::waitKey(10);

	//return output;
}
void Kinect2::Add_Grid_Points(std::vector<pair<int, int>> *observed_points, cv::Mat *Point_Grid)
{
	int number_of_neighborhood = 3 * 255;
	for (int rows = 1; rows < Point_Grid->rows - 1; rows++)	
	{
		for (int cols = 1; cols < Point_Grid->cols - 1; cols++)
		{
			if (Point_Grid->at<unsigned char>(rows, cols) == 0)
				continue;

			int neighborhood_flag = 0;

			neighborhood_flag += Point_Grid->at<unsigned char>(rows - 1, cols);
			neighborhood_flag += Point_Grid->at<unsigned char>(rows + 1, cols);
			neighborhood_flag += Point_Grid->at<unsigned char>(rows, cols - 1);
			neighborhood_flag += Point_Grid->at<unsigned char>(rows, cols + 1);

			neighborhood_flag += Point_Grid->at<unsigned char>(rows - 1, cols-1);
			neighborhood_flag += Point_Grid->at<unsigned char>(rows + 1, cols-1);
			neighborhood_flag += Point_Grid->at<unsigned char>(rows-1, cols + 1);
			neighborhood_flag += Point_Grid->at<unsigned char>(rows+1, cols + 1);

			if (neighborhood_flag>number_of_neighborhood)
			{
				observed_points->push_back(pair<int, int>(cols, rows));
			}
				
		}
	}
}
double Kinect2::get_gapTH(double from, double to)
{
	if (from == 0 && to == 0)
		return 0;

	double gap_theta_getTH = 0;
	double sign_from = from / abs(from);
	double sign_to = to / abs(to);
	if (sign_from + sign_to == 0)
	{
		gap_theta_getTH = sign_to*(abs(from) + abs(to));
		if (abs(gap_theta_getTH)>180)
			gap_theta_getTH = (gap_theta_getTH - gap_theta_getTH / abs(gap_theta_getTH) * 360);
	}
	else
		gap_theta_getTH = to - from;

	return gap_theta_getTH;
}
cv::Mat Kinect2::drawPoints(std::map<int, int> *observed_points)
{
	cv::Mat output_img(480, 300, CV_8UC1, cv::Scalar(0, 0, 0));
	for (std::map<int, int>::iterator iter = observed_points->begin(); iter != observed_points->end(); iter++)
		cv::circle(output_img, cv::Point(iter->first, iter->second), 1, cv::Scalar(255), -1);
	
	return output_img;
}
cv::Mat Kinect2::drawPoints(cv::Mat *points, cv::Mat *RT_mat)
{
	cv::Mat output_img(480, 300, CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat temp_points(3, 1, CV_64FC1, cv::Scalar(1));
	cv::Mat output_point(2, 1, CV_64FC1);
	for (int i = 0; i < points->rows; i++)
	{
		temp_points.at<double>(0, 0) = points->row(i).at<double>(0, 0);
		temp_points.at<double>(1, 0) = points->row(i).at<double>(0, 1);

		output_point = *RT_mat*temp_points;

		cv::circle(output_img, cv::Point(output_point.at<double>(0, 0), output_point.at<double>(1, 0)), 1, cv::Scalar(255), -1);
	}
	return output_img;
}
cv::Mat Kinect2::drawPoints(cv::Mat *points, cv::Mat *RT_mat, cv::Mat *output_img)
{
	//cv::Mat output_img(480, 300, CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat temp_points(3, 1, CV_64FC1, cv::Scalar(1));
	cv::Mat output_point(2, 1, CV_64FC1);
	for (int i = 0; i < points->rows; i++)
	{
		temp_points.at<double>(0, 0) = points->row(i).at<double>(0, 0);
		temp_points.at<double>(1, 0) = points->row(i).at<double>(0, 1);

		output_point = *RT_mat*temp_points;

		cv::circle(*output_img, cv::Point(output_point.at<double>(0, 0), output_point.at<double>(1, 0)), 1, cv::Scalar(0), -1);
	}
	return *output_img;
}
double Kinect2::get_Rotation_from_mat(cv::Mat *RTmat)
{
	double cos_val = RTmat->at<double>(0,0);
	double sin_val = -RTmat->at<double>(0,1);
	double outputangle;	

	if (cos_val < 0)
	{
		outputangle = 180.0-180.0*asin(sin_val) / CV_PI;
		if (sin_val < 0)
			outputangle -= 360.0;
	}
	else
	{
		outputangle = 180.0*asin(sin_val) / CV_PI;
	}
	return outputangle;
}
double Kinect2::initial_rotation_matcher(cv::Mat *from, cv::Mat *to, LocalGrid *Gridinfo, double *initial_gauss)
{
	cv::imshow("to", *to);
	cv::imshow("from", *from);

	double test_range=40.0;
	double test_angle_gap = 2.0;

	cv::Mat rota_Mat;
	cv::Mat temp_rotated_from;

	cv::Mat preprocessing_toMat(to->rows * 2, to->rows * 2, CV_8UC1, cv::Scalar(0));
	int img_tran[2] = { (preprocessing_toMat.cols - to->cols) / 2, (preprocessing_toMat.rows - to->rows) / 2 };
	cv::Mat ROI_toMat = preprocessing_toMat(Rect(img_tran[0], img_tran[1], to->cols, to->rows));
	to->copyTo(ROI_toMat);
	cv::resize(preprocessing_toMat, preprocessing_toMat, cv::Size(preprocessing_toMat.cols / 2, preprocessing_toMat.rows / 2));
	cv::GaussianBlur(preprocessing_toMat, preprocessing_toMat, cv::Size(11, 11), 0);

	cv::imshow("preprocessing_toMat", preprocessing_toMat);

	double minimum_cost=99999.0;
	double minimum_angle = initial_gauss[2];
	for (float test_angle = initial_gauss[2] - test_range; test_angle < initial_gauss[2] + test_range; test_angle += test_angle_gap)
	{
		rota_Mat = cv::getRotationMatrix2D(cv::Point2f(Gridinfo->robot_posi_in_cols, Gridinfo->robot_posi_in_rows), test_angle,0.5);
		rota_Mat.at<double>(0, 2) += (initial_gauss[0] + img_tran[0]/2);
		rota_Mat.at<double>(1, 2) += (initial_gauss[1] + img_tran[1]/2);
		cv::warpAffine(*from, temp_rotated_from, rota_Mat, cv::Size(preprocessing_toMat.cols, preprocessing_toMat.rows), 1, 0, cv::Scalar(0));
		cv::imshow("warpAffine", temp_rotated_from);
		cv::waitKey(30);
		//sum_mat = max(rotated_b_local_grid, target);
	}

	return 0.0;
}