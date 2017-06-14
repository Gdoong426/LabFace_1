
// LabFaceDlg.cpp : implementation file
//

#include "stdafx.h"
#include "LabFace.h"
#include "LabFaceDlg.h"
#include "afxdialogex.h"

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>



#include <dlib\opencv.h>
#include <dlib\data_io.h>
#include <dlib\image_processing.h>
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing\generic_image.h>
#include <dlib\gui_widgets.h>
#include <dlib\image_io.h>

#include <stdio.h>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CLabFaceDlg dialog



CLabFaceDlg::CLabFaceDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_LABFACE_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CLabFaceDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CLabFaceDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CLabFaceDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CLabFaceDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CLabFaceDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CLabFaceDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CLabFaceDlg::OnBnClickedButton5)
END_MESSAGE_MAP()


// CLabFaceDlg message handlers

BOOL CLabFaceDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	AllocConsole();
	freopen("CONOUT$", "w", stdout);

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CLabFaceDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CLabFaceDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CLabFaceDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

string name_0 = "Dong_0";
string name_1 = "Rain_1";
string name_2 = "Lewis_2";
string name_3 = "Daniel_3";
string name_4 = "Chi Ching_4";
string name_5 = "Torbjorn Nordling_5";
string name_6 = "Tim_6";

void CLabFaceDlg::OnBnClickedButton1()
{
	// Detect faces in current frame

	// Open webcam
	VideoCapture cap = (-1);

	// Load in Haar filter that detect faces
	String face_cascade_name = "D:/VS 2015 Project/LabFace_1/LabFace/haarcascade_frontalface_default.xml";
	CascadeClassifier face_cascade;
	string window_name = "face_detection";
	if (!face_cascade.load(face_cascade_name)) cout << "error loading face cascade file..." << endl;
	

	int num_components = 10; // Number of Eigenfaces 
	//double threshold = 0; // Set threshold for face recognizer 
	Ptr<FaceRecognizer> model1 = createEigenFaceRecognizer(num_components);
	// Load the eigenfaces file created in trainning section.
	model1->load("D:/VS 2015 Project/LabFace_1/LabFace/eigenfaces_at.yml");
	Mat mean = model1->getMat("mean");

	/*Ptr<FaceRecognizer> model2 = createFisherFaceRecognizer();
	model2->load("fisherfaces_at.yml");

	Ptr<FaceRecognizer> model3 = createLBPHFaceRecognizer();
	model3->load("LBPHF_at.yml");*/

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("D:/VS 2015 Project/LabFace_1/LabFace/shape_predictor_68_face_landmarks.dat") >> pose_model;
	int frm_num = 0;
	
	
	Mat frame;
	Mat face_image;
	cap >> frame;
	while (frame.empty() != 1) {

		// load video frame into opencv Mat structure
		cap >> frame;
		Mat original = frame.clone();
		vector< Rect_<int>> faces;
		Mat frame_gray;
		cvtColor(original, frame_gray, CV_RGB2GRAY);
		equalizeHist(frame_gray, frame_gray);

		// Face detection by haar filter.
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
		if (faces.size() > 0) {
			for (size_t i = 0; i < faces.size(); i++) {

				//Rect face_i = faces[i];
				Mat face = frame_gray(faces[i]).clone();
				Mat face_resize;
				resize(face, face_resize, Size(mean.cols, mean.rows), 1.0, 1.0, INTER_CUBIC);
				string name;

				// prediction of the faces in the video.
				int prediction = model1->predict(face_resize);
				switch (prediction) {
				case 0:
					name = name_0;	break;
				case 1:
					name = name_1;	break;
				case 2:
					name = name_2;	break;
				case 3:
					name = name_3;	break;
				case 4:
					name = name_4;	break;
				case 5:
					name = name_5;	break;
				case -1:
					name = "No one..";
				}


				// Resize the size of rectangle to fit faces.
				int w_temp = faces[i].width;
				int h_temp = faces[i].height;
				int x = faces[i].x - w_temp*0.05;
				int y = faces[i].y - h_temp*0.2;

				if (x < 0) x = 0;
				if (y < 0) y = 0;
				
				int w = faces[i].width*1.1;
				int h = h_temp*1.4;
				Rect biggerRect(x, y, w, h);

				Mat face_image;
				if (0 <= biggerRect.x
					&& 0 <= biggerRect.width
					&& biggerRect.x + biggerRect.width <= original.cols
					&& 0 <= biggerRect.y
					&& 0 <= biggerRect.height
					&& biggerRect.y + biggerRect.height <= original.rows){
					
					face_image = original(biggerRect).clone();
					imshow("face_img", face_image);
				}

				else {
					original.copyTo(face_image);
				}
				//rectangle(original, biggerRect, Scalar(255, 255, 0), 2);
				

				dlib::cv_image <dlib::bgr_pixel> cimg(face_image);
				std::vector<dlib::rectangle> faces_dlib = detector(cimg);
				std::vector < dlib::full_object_detection> shapes;

				// Use dlib to find faces and its pose from cropped face_image
				for (unsigned long j = 0; j < faces_dlib.size(); ++j) {
					frm_num++;
					shapes.push_back(pose_model(cimg, faces_dlib[j]));
					dlib::array<dlib::array2d<dlib::bgr_pixel>>face_chips;
					
					// extract face image with eyes alignment
					extract_image_chips(cimg, get_face_chip_details(shapes), face_chips); 
					Mat faces_crop;
					faces_crop = toMat(face_chips[j]);
					imshow("detected", faces_crop);

					// !!!Please change to the correct lab members' name!!!
					string face_str = "./Face_croped/F";
					string file_type = ".jpg";
					char face_char[21];
					char time_str[22];

					// find the current time/ date based on current system and transfer to string
					time_t now = time(0);
					struct tm * timeinfo;
					time(&now);
					timeinfo = localtime(&now);

					strftime(time_str, sizeof(time_str), "(%m%d_%I%M)", timeinfo);
					string str(time_str);

					face_str = face_str + itoa(frm_num, face_char, 10) + str + file_type;
					cout << face_str << endl;



					// Save the cropped image for latter trainning
					imwrite(face_str, faces_crop);
					rectangle(original, faces[i], CV_RGB(255, 0, 0), 1);

					// name tags position in video
					int pos_x = max(faces[i].tl().x - 10, 0);
					int pos_y = max(faces[i].tl().y - 10, 0);

					putText(original, name, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
				}

			}

		}

		imshow(window_name, original);
		char c = waitKey(30);
		if (c == 27) {
			destroyWindow(window_name);
			break;
		}
	}
}




void CLabFaceDlg::OnBnClickedButton2()
{	

	// find faces and its pose in order to crop the face image with eyes alignment.
	using namespace dlib;

	Mat frame;
	VideoCapture cap(-1);


	dlib::image_window win, win_faces;
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	// using dlib build-in dat file to detect face pose
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	int frm_num = 0;
	while (1) {
		// start video frame
		cap >> frame;

		imshow("frame", frame);
		frm_num++;
		cv_image <bgr_pixel> cimg(frame);

		// ask the face detector to give us a list of bounding boxes
		// around all the faces in the image
		std::vector<dlib::rectangle> faces = detector(cimg);

		// ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector < full_object_detection> shapes;
		
		for (unsigned long i = 0; i < faces.size(); i++) {
			shapes.push_back(pose_model(cimg, faces[i]));

			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));

			dlib::array<array2d<bgr_pixel>>face_chips;
			extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
			win_faces.set_image(tile_images(face_chips));
			
			// transfer dlib image to opencv Mat structure
			Mat faces;
			faces = toMat(face_chips[i]);
			imshow("detected", faces);


			// !!!Please change to the correct lab members' name!!!
			string face_str = "./Face_croped/Face";
			string file_type = ".jpeg";
			char face_char[21];
			char time_str[22];

			// find the current time/ date based on current system and transfer to string
			time_t now = time(0);
			struct tm * timeinfo;
			time(&now);
			timeinfo = localtime(&now);

			strftime(time_str, sizeof(time_str), "(%m-%d-%Y_%I:%M:%S)", timeinfo);
			string str(time_str);

			face_str = face_str + itoa(frm_num, face_char, 10) + str + file_type;
			cout << face_str << endl;
		
			
			
		
			imwrite(face_str, faces);
			
		}

		char cmd = waitKey(33);
		if (cmd == 27) {
			cv::destroyWindow("frame");
		}
	}
}

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


void CLabFaceDlg::OnBnClickedButton3()
{

	// Using multiple face trainning methods to traing its filter.
	string fn_csv = "TotalFaceList.csv";
	vector<Mat> images;
	vector<int> labels;

	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		waitKey(0);
	}

	if (images.size() <= 1) {
		string error_msg = "Please input more than one images...";
		CV_Error(CV_StsError, error_msg);
	}

	int heiget = images[0].rows;
	

	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	imshow("image1", images[0]);
	imshow("image2", testSample);
	waitKey(0);

	int recognizer = 1;
	cout << "choose a face recognizer: 1. EigenFace, 2. FisherFace, 3. Local Binary Patterns " << endl;
	//cin >> recognizer;
	//getline(cin, recognizer, '\n');

	switch (recognizer)
	{
	//case 1:

	default:
	{
		cout << recognizer << endl;
		Ptr<FaceRecognizer> model = createEigenFaceRecognizer();

		model->train(images, labels);
		model->save("D:/VS 2015 Project/LabFace_1/LabFace/eigenfaces_at.yml");

		int predictedLabel = model->predict(testSample);

		string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
		cout << result_message << endl;

		Mat eigenValues = model->getMat("eigenvalues");

		Mat W = model->getMat("eigenvectors");

		Mat mean = model->getMat("mean");

		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));

		waitKey(80);

		for (int i = 0; i < min(10, W.cols); i++) {
			Mat ev = W.col(i).clone();

			Mat Grayscale = norm_0_255(ev.reshape(1, heiget));
			Mat cGrayscale;
			applyColorMap(Grayscale, cGrayscale, COLORMAP_JET);

			imshow("eigenface", cGrayscale);
			imwrite(format("./eigenFaces/eigenFaces_%d.jpg", i), norm_0_255(cGrayscale));
			waitKey(100);
		}

		for (int num_components = 10; num_components < 300; num_components += 15) {
			// slice the eigenvectors from the model
			Mat evs = Mat(W, Range::all(), Range(0, num_components));
			Mat projection = subspaceProject(evs, mean, images[0].reshape(1, 1));
			Mat reconstruction = subspaceReconstruct(evs, mean, projection);
			// Normalize the result:
			reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));


			imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
			imwrite(format("./eigenFaces_reconst/eigenface_reconstruction_%d.png", num_components), reconstruction);
			waitKey(100);
		}	
		cout << "Finish eigenfaces trainning..." << endl;
	}
	break;
	case 2:
	{
		cout << "%d" << recognizer << endl;
		Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
		model->train(images, labels);
		model->save("fisherfaces_at.yml");
		cout << "Finish fisherfaces trainning..." << endl;
	}
		break;
	case 3:
	
	{
		cout << "%d" << recognizer << endl;
		Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
		model->train(images, labels);
		model->save("LBPHF_at.yml");
		cout << "Finish LBPHF trainning..." << endl;
	}
		break;


	}
		
	

	
}


void CLabFaceDlg::OnBnClickedButton4()
{


	// load single images in /Face/ folder and its csv file to create face samples
	Mat frame;
	string fn_csv = "./Face/FaceList.csv";
	vector<Mat> images;
	vector<int> labels;

	

	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cout << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		waitKey(0);
	}
	cout << images.size() << endl;
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	int frm_num = 0;
	for (int frm_num = 0; frm_num< labels.size(); frm_num++){


		dlib::cv_image <dlib::bgr_pixel> cimg(images[frm_num]);

		// ask the face detector to give us a list of bounding boxes
		//around all the faces in the image
		std::vector<dlib::rectangle> faces = detector(cimg);

		//ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector < dlib::full_object_detection> shapes;
		imshow("frame", images[frm_num]);
		for (unsigned long i = 0; i < faces.size(); i++) {
			shapes.push_back(pose_model(cimg, faces[i]));

			dlib::array<dlib::array2d<dlib::bgr_pixel>>face_chips;
			extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
			Mat faces;
			faces = toMat(face_chips[i]);
			imshow("detected", faces);


			// !!!Please change to the correct lab members' name!!!
			string face_str = "./Face_croped/Face";
			string file_type = ".jpeg";
			char face_char[21];
			face_str = face_str + itoa(frm_num, face_char, 10) + file_type;
			//cout << face_str << endl;

			


			imwrite(face_str, faces);

		}
		waitKey(100);
	}
}


void CLabFaceDlg::OnBnClickedButton5()
{
	// TODO: Add your control notification handler code here
}
