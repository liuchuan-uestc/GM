package Classification;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import Classification.CBC;
import Classification.SVM;
import PreProcess.DataPreProcess;

public class Test{
	
	public static final int CBCindex = 1;
	public static final int GMindex  = 2;
	public static final int DPindex  = 3;
	public static final int SVMindex = 4;
	public static final int endindex = 5;
	
	public static void main(String arg[]){
		try {
			MainMethod();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
	}
	
	public static void MainMethod() throws IOException {

	int FilterMapNum = 4;
	int TrainTestNum = 5; 
	double accuracy  = 0;
	String StrSrcDir = "./DataMiningSample/";
	String StrSrcDirData = "./DataMiningSample/3_TrainData";
	Map<Integer, Double> IndexValueMap = new TreeMap<Integer, Double>();
	Map<Integer, Double> IndexValueMapNew = new TreeMap<Integer, Double>();
	Map<Integer, Double> IndexDisValueMap = new TreeMap<Integer, Double>();
	Map<Integer, Double> IndexDisValueMapNew = new TreeMap<Integer, Double>();
	Map<String, Double> WordTFMap = new TreeMap<String, Double>();
	Map<String, Map<Integer, Double>> trainFileMapVsm  = new TreeMap<String, Map<Integer, Double>>();
	Map<String, Map<Integer, Double>> testFileMapVsm   = new TreeMap<String, Map<Integer, Double>>();
	Map<String, Map<Integer, Double>> trainMapNew  = new TreeMap<String, Map<Integer, Double>>();
	Map<Integer,String> classes  = new TreeMap<Integer, String>();
	Map<String,String> pred   = new TreeMap<String, String>();
	Map<String,String> actual = new TreeMap<String, String>();
	Map<String, Double> TempResultMap = new TreeMap<String, Double>();
	
	Map<Integer, Double> accuracyMap = new TreeMap<Integer, Double>();
	
	DataPreProcess     DataPP = new DataPreProcess();
	
	//DataPreProcess.SetPrintOut(StrSrcDir);
	System.out.println("Test start...");

	for(int i=1; i<endindex; i++)
	{
		String StrSrcDir_Method = "";
		String StrSrcDir_Method_output = "";
		
		double[] TempResult_macroF1 = new double[TrainTestNum+1];
		double[] TempResult_microF1 = new double[TrainTestNum+1];
		double[] TempResult_Entropy = new double[TrainTestNum+1];

		switch(i)
		{
			case CBCindex:
				StrSrcDir_Method = StrSrcDir+"CBC/";
				break;
			case GMindex:
				StrSrcDir_Method = StrSrcDir+"GM/";
				break;
			case DPindex:
				StrSrcDir_Method = StrSrcDir+"DP/";
				break;
			case SVMindex:
				StrSrcDir_Method = StrSrcDir+"SVM/";
				break;
			default:
				System.err.println("No such method...");
				break;
		}
		
		StrSrcDir_Method_output = StrSrcDir_Method+"0_outputfile/";
		DataPP.CreatDir(StrSrcDir_Method);
		DataPP.CreatDir(StrSrcDir_Method_output);
		
		for(int j=0; j<TrainTestNum; j++)
		{
				String VsmTestSrcDir  = StrSrcDirData+"/4_DocVector/VsmTFIDFMapTestSample"+j+".txt";
				String VsmTrainSrcDir = StrSrcDirData+"/4_DocVector/VsmTFIDFMapTrainSample"+j+".txt";

				testFileMapVsm  = DataPP.ReadFileVsm(VsmTestSrcDir);
				trainFileMapVsm = DataPP.ReadFileVsm(VsmTrainSrcDir);

				Map<String, Map<Integer, Double>> temptrainFileMapVsm = new TreeMap<String, Map<Integer, Double>>();
				
				if(true){
					if(i==CBCindex){
						System.out.println("CBC start...");
						actual.clear();
						pred.clear();
						classes.clear();
						temptrainFileMapVsm = DataPP.copyMaptoMap(trainFileMapVsm); 
			
						accuracy = CBC.CBCMain(StrSrcDir_Method_output,temptrainFileMapVsm,testFileMapVsm,actual,pred);
						
						String DesFile = StrSrcDir_Method_output+"CBC_"+i+"_"+j+".txt";
			
						TempResultMap = DataPP.compute_accuracy_F_RetMap(actual, pred, classes, DesFile, 2, 1);
			
						TempResult_macroF1[j] = TempResultMap.get("macro_F1");
						TempResult_microF1[j] = TempResultMap.get("micro_F1");
						TempResult_Entropy[j] = TempResultMap.get("Entropy");
					}
					
					if(i==GMindex){
						System.out.println("GM start...");
						actual.clear();
						pred.clear();
						classes.clear();
						temptrainFileMapVsm = DataPP.copyMaptoMap(trainFileMapVsm); 
						
						accuracy = GM.GMMain(StrSrcDir_Method_output,temptrainFileMapVsm,testFileMapVsm,actual,pred);
						
						String DesFile = StrSrcDir_Method_output+"GM_"+i+"_"+j+".txt";
			
						TempResultMap = DataPP.compute_accuracy_F_RetMap(actual, pred, classes, DesFile, 2, 1);
			
						TempResult_macroF1[j] = TempResultMap.get("macro_F1");
						TempResult_microF1[j] = TempResultMap.get("micro_F1");
						TempResult_Entropy[j] = TempResultMap.get("Entropy");
					}
					if(i==DPindex){
						System.out.println("DP start...");
						actual.clear();
						pred.clear();
						classes.clear();
						temptrainFileMapVsm = DataPP.copyMaptoMap(trainFileMapVsm); 
						
						accuracy = DP.DPMain(StrSrcDir_Method_output,temptrainFileMapVsm,testFileMapVsm,actual,pred);
						
						String DesFile = StrSrcDir_Method_output+"DP_"+i+"_"+j+".txt";
			
						TempResultMap = DataPP.compute_accuracy_F_RetMap(actual, pred, classes, DesFile, 2, 1);
			
						TempResult_macroF1[j] = TempResultMap.get("macro_F1");
						TempResult_microF1[j] = TempResultMap.get("micro_F1");
						TempResult_Entropy[j] = TempResultMap.get("Entropy");
					}
					
					if(i==SVMindex){
						System.out.println("SVM start...");
						actual.clear();
						pred.clear();
						classes.clear();
						temptrainFileMapVsm = DataPP.copyMaptoMap(trainFileMapVsm); 
						
						accuracy = SVM.SVMMain(StrSrcDir_Method_output,temptrainFileMapVsm,testFileMapVsm,actual,pred,i,j);
						
						String DesFile = StrSrcDir_Method_output+"SVM_"+i+"_"+j+".txt";
			
						TempResultMap = DataPP.compute_accuracy_F_RetMap(actual, pred, classes, DesFile, 2, 1);
			
						TempResult_macroF1[j] = TempResultMap.get("macro_F1");
						TempResult_microF1[j] = TempResultMap.get("micro_F1");
						TempResult_Entropy[j] = TempResultMap.get("Entropy");
					}
				}	
	
		}
	
			for(int m=0; m<TrainTestNum; m++)
			{
				TempResult_macroF1[TrainTestNum] += TempResult_macroF1[m];
				TempResult_microF1[TrainTestNum] += TempResult_microF1[m];
				TempResult_Entropy[TrainTestNum] += TempResult_Entropy[m];
			}
	
			TempResult_macroF1[TrainTestNum] = TempResult_macroF1[TrainTestNum]/5.0;
			TempResult_microF1[TrainTestNum] = TempResult_microF1[TrainTestNum]/5.0;
			TempResult_Entropy[TrainTestNum] = TempResult_Entropy[TrainTestNum]/5.0;
		
			Map<String, Map<Integer, String>> RESULTMAP = new TreeMap<String, Map<Integer, String>>();
			Map<Integer,String> IndexMapNewNumMap  = new TreeMap<Integer, String>();
			Map<Integer,String> TempResult_macroF1Map  = new TreeMap<Integer, String>();
			Map<Integer,String> TempResult_microF1Map  = new TreeMap<Integer, String>();
			Map<Integer,String> TempResult_EntropyMap  = new TreeMap<Integer, String>();
	
			for(int m=TrainTestNum; m<=TrainTestNum; m++)
			{
				TempResult_macroF1Map.put(m+1, DataPP.fixedWidthIntegertoSpace(""+TempResult_macroF1[m],20));
				TempResult_microF1Map.put(m+1, DataPP.fixedWidthIntegertoSpace(""+TempResult_microF1[m],20));
				TempResult_EntropyMap.put(m+1, DataPP.fixedWidthIntegertoSpace(""+TempResult_Entropy[m],20));

				System.out.println("=============================================");
				System.out.println("TempResult_macroF1 :"+DataPP.fixedWidthIntegertoSpace(""+TempResult_macroF1[m],20));
				System.out.println("TempResult_microF1 :"+DataPP.fixedWidthIntegertoSpace(""+TempResult_microF1[m],20));
				System.out.println("TempResult_Entropy :"+DataPP.fixedWidthIntegertoSpace(""+TempResult_Entropy[m],20));	
			}
	
			RESULTMAP.put("001_TempResult_macroF1Map", TempResult_macroF1Map);
			RESULTMAP.put("002_TempResult_microF1Map", TempResult_microF1Map);
			RESULTMAP.put("003_TempResult_EntropyMap", TempResult_EntropyMap);
	
			String FileName = "Classification_Method_"+i+".txt";
			DataPP.printSISMapForLW(StrSrcDir_Method_output, FileName , RESULTMAP);	

		}
		System.out.println("Test Finished!!!");
	}
}
