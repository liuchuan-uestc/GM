package Classification;

import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import SVM.svm_predict;
import SVM.svm_train;
import SVM.libsvm.svm_model;

public class GM{

	public static double GMMain(String strDir, Map<String, Map<Integer, Double>> TrainFileMapVsm, 
								           Map<String, Map<Integer, Double>> TestFileMapVsm,
								           Map<String,String> actual, Map<String,String> pred) throws IOException {

		double accuracy = GMFactor(strDir, TrainFileMapVsm, TestFileMapVsm, actual, pred);
		return accuracy;
	}

	public static double GMFactor(String strDir, Map<String, Map<Integer, Double>> TrainFileMapVsm, 
								           Map<String, Map<Integer, Double>> TestFileMapVsm,
								           Map<String,String> actual, Map<String,String> pred) throws IOException {
		//训练集中相同的类放在一个map中，形成同类样本map
		int  Tsize = TrainFileMapVsm.size();
		Map<String, Map<String, Integer>> CateMap = new TreeMap<String, Map<String, Integer>>();
		Map<String, Map<Integer, Double>> CateCenterMap = new TreeMap<String, Map<Integer, Double>>();

		Set<Map.Entry<String, Map<Integer, Double>>> TrainFileMapVsmSet = TrainFileMapVsm.entrySet();
		for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = TrainFileMapVsmSet.iterator(); it.hasNext();) {
			Map.Entry<String, Map<Integer, Double>> me = it.next();
			
			String Cate = me.getKey().split("_")[0];
			if(CateMap.containsKey(Cate)){
				if(CateMap.get(Cate).containsKey(me.getKey())){
					int count = CateMap.get(Cate).get(me.getKey());
					CateMap.get(Cate).put(me.getKey(), count+1);
				}
				else{
					CateMap.get(Cate).put(me.getKey(),1);
				}
			}
			else{
			    Map<String, Integer> tempMap = new TreeMap<String, Integer>();
				tempMap.put(me.getKey(),1);
				CateMap.put(Cate, tempMap);
			}
		}

        //printSSIMap(strDir, "CateMap.txt", CateMap);
		//寻找每个类别的质心点
		int Ksize = 0;
		int j = 0;
		int i = 0;
		int[] CateNum = new int[CateMap.size()];

		Set<Map.Entry<String, Map<String, Integer>>> CateMapSet = CateMap.entrySet();
		for (Iterator<Map.Entry<String, Map<String, Integer>>> it = CateMapSet.iterator(); it.hasNext();) {
			Map.Entry<String, Map<String, Integer>> me = it.next();
			CateNum[i++] = me.getValue().size();
			Set<Map.Entry<String, Integer>> allVsmSet2 = me.getValue().entrySet();
			for (Iterator<Map.Entry<String, Integer>> it2 = allVsmSet2.iterator(); it2.hasNext();) {
				Map.Entry<String, Integer> me2 = it2.next();
				if(CateCenterMap.containsKey(me.getKey())){
					Set<Map.Entry<Integer, Double>> allVsmSet3 = TrainFileMapVsm.get(me2.getKey()).entrySet();
					for (Iterator<Map.Entry<Integer, Double>> it3 = allVsmSet3.iterator(); it3.hasNext();) {
						Map.Entry<Integer, Double> me3 = it3.next();
						if(CateCenterMap.get(me.getKey()).containsKey(me3.getKey())){
							double p = CateCenterMap.get(me.getKey()).get(me3.getKey());
							CateCenterMap.get(me.getKey()).put(me3.getKey(),p+me3.getValue());
						}else{
							CateCenterMap.get(me.getKey()).put(me3.getKey(), me3.getValue());
						}
					}
				}
				else{
					Map<Integer, Double> tempMap = new TreeMap<Integer, Double>();
					Set<Map.Entry<Integer, Double>> allVsmSet3 = TrainFileMapVsm.get(me2.getKey()).entrySet();
					for (Iterator<Map.Entry<Integer, Double>> it3 = allVsmSet3.iterator(); it3.hasNext();) {
						Map.Entry<Integer, Double> me3 = it3.next();
						tempMap.put(me3.getKey(), me3.getValue());
					}
					if(Ksize<tempMap.size()){
						Ksize = tempMap.size();
					}
					CateCenterMap.put(me.getKey(), tempMap);
				}			
			}
		}
        //printSIDMap(strDir, "CateCenterMap.txt", CateCenterMap);

		i = 0;
		String[] CateName = new String[CateCenterMap.size()];
		Set<Map.Entry<String, Map<Integer, Double>>> CateCenterMapSet = CateCenterMap.entrySet();
		for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = CateCenterMapSet.iterator(); it.hasNext();) {
			Map.Entry<String, Map<Integer, Double>> me = it.next();
			CateName[i] = me.getKey();
			Set<Map.Entry<Integer, Double>> allVsmSet2 = me.getValue().entrySet();
			for (Iterator<Map.Entry<Integer, Double>> it2 = allVsmSet2.iterator(); it2.hasNext();) {
				Map.Entry<Integer, Double> me2 = it2.next();

				CateCenterMap.get(me.getKey()).put(me2.getKey(),me2.getValue()/CateNum[i]);
			}
			i++;
		}
		Map<String, Double> CateCenterFactorMap = new TreeMap<String, Double>();
		CateCenterFactorMap = AdjustFactorForCentriod(TrainFileMapVsm, TestFileMapVsm, CateCenterMap);
		TestResult(CateCenterFactorMap,TestFileMapVsm,CateCenterMap, actual, pred);
		return 0;
	}

    //试验每个图心给予不同的半径
	public static Map<String, Double> AdjustFactorForCentriod(Map<String, Map<Integer, Double>> TrainFileMapVsm,
															  Map<String, Map<Integer, Double>> TestFileMapVsm,
								          		              Map<String, Map<Integer, Double>> CateCenterMap) throws IOException {
		
		Map<String, Double> CateCenterFactorMap = new TreeMap<String, Double>();
		Set<Map.Entry<String, Map<Integer, Double>>> CateCenterMapSet = CateCenterMap.entrySet();
		for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = CateCenterMapSet.iterator(); it.hasNext();) {
			Map.Entry<String, Map<Integer, Double>> me = it.next();
			CateCenterFactorMap.put(me.getKey(),1.0);
		}

		int i = 0;
		int j = 0;
		double rate   = 0.0;
		double factor = 0.0;
		double learnrate = 0.001;
		double Newaccray = 0.001;
		double Oldaccray = 0.001;
		double lift = 1.0;
		double[][] TrainDistance = new double[TrainFileMapVsm.size()][CateCenterMap.size()+1];

		for(int n=0; ((n<6)||(lift>0.001)&&(n<50)); n++)
		{
			Oldaccray = Newaccray;
			System.out.println("----------------First step "+n+", lift: "+lift+", learnrate: "+learnrate*rate+"----------------");
			for(i=0; i<TrainFileMapVsm.size(); i++){
				for(j=0; j<CateCenterMap.size()+1; j++){
					TrainDistance[i][j] = 0xFFFFFF;
				}
			}
			
			i = 0;
			j = 0;
			Set<Map.Entry<String, Map<Integer, Double>>> TrainFileMapVsmSet = TrainFileMapVsm.entrySet();
			for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = TrainFileMapVsmSet.iterator(); it.hasNext();) {
				Map.Entry<String, Map<Integer, Double>> me = it.next();
				j = 0;
				Set<Map.Entry<String, Map<Integer, Double>>> CateCenterMapSet1 = CateCenterMap.entrySet();
				for (Iterator<Map.Entry<String, Map<Integer, Double>>> it2 = CateCenterMapSet1.iterator(); it2.hasNext();) {
					Map.Entry<String, Map<Integer, Double>> me2 = it2.next();
					j = Integer.parseInt(me2.getKey());	
					TrainDistance[i][j] = getDistance2(CateCenterFactorMap.get(me2.getKey()), me.getValue(),me2.getValue());
				}
			    i++;
			}

			int[] nearestMeans = new int[TrainFileMapVsm.size()];
			for (i = 0; i < TrainFileMapVsm.size(); i++) {
				nearestMeans[i] = findNearestMeans(TrainDistance, i);
			}

			i = 0;
			int correct = 0;
			int total   = TrainFileMapVsm.size();
			Set<Map.Entry<String, Map<Integer, Double>>> TrainFileMapVsmSet2 = TrainFileMapVsm.entrySet();
			for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = TrainFileMapVsmSet2.iterator(); it.hasNext();) {
				Map.Entry<String, Map<Integer, Double>> me = it.next();
				String Cate = me.getKey().split("_")[0];
				//System.out.println("i: "+i+", Cate: "+Integer.parseInt(Cate)+", nearestMeans: "+nearestMeans[i]);
				if(Integer.parseInt(Cate) == nearestMeans[i]){
					correct += 1.0;
				}
				else{
					//rate = TrainDistance[i][Integer.parseInt(Cate)] - TrainDistance[i][nearestMeans[i]];
					rate = 0.1;
					factor = CateCenterFactorMap.get(Cate);
					factor = factor + learnrate*rate;
					CateCenterFactorMap.put(Cate, factor);
					
					Cate = ""+nearestMeans[i];
					factor = CateCenterFactorMap.get(Cate);
					factor = factor - learnrate*rate;
					CateCenterFactorMap.put(Cate, factor);
				}
				i++;
			}
			System.out.println("Train Accuracy = "+(double)correct/total*100+"% ("+correct+"/"+total+") (AdjustFactorForCentriod Classifier)\n");
			TestResult(CateCenterFactorMap, TestFileMapVsm, CateCenterMap);
			if(true)
			{
				System.out.print("Factor:");
				Set<Map.Entry<String, Map<Integer, Double>>> CateCenterMapSet2 = CateCenterMap.entrySet();
				for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = CateCenterMapSet2.iterator(); it.hasNext();) {
					Map.Entry<String, Map<Integer, Double>> me = it.next();
					System.out.print(" "+me.getKey()+":"+CateCenterFactorMap.get(me.getKey()));
				}	
				System.out.println(" ");
				System.out.println(" ");
			}
			
			Newaccray = (double)correct/(double)total;
			lift = (Newaccray-Oldaccray)/Oldaccray;
			
		}
		return CateCenterFactorMap;
	}

	public static double TestResult(Map<String, Double> CateCenterFactorMap,
									Map<String, Map<Integer, Double>> TestFileMapVsm,
								    Map<String, Map<Integer, Double>> CateCenterMap,
								    Map<String,String> actual, Map<String,String> pred) throws IOException {
		int i = 0;
		int j = 0;
		double[][] TestDistance = new double[TestFileMapVsm.size()][CateCenterMap.size()+1];

		//System.out.println("MultlinearMap.size(): "+MultlinearMap.size());
		for(i=0; i<TestFileMapVsm.size(); i++){
			for(j=0; j<CateCenterMap.size()+1; j++){
				TestDistance[i][j] = 10.0;
			}
		}
		
		i = 0;
		j = 0;
		Set<Map.Entry<String, Map<Integer, Double>>> TestFileMapVsmSet = TestFileMapVsm.entrySet();
		for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = TestFileMapVsmSet.iterator(); it.hasNext();) {
			Map.Entry<String, Map<Integer, Double>> me = it.next();
			j = 0;
			Set<Map.Entry<String, Map<Integer, Double>>> CateCenterMapSet1 = CateCenterMap.entrySet();
			for (Iterator<Map.Entry<String, Map<Integer, Double>>> it2 = CateCenterMapSet1.iterator(); it2.hasNext();) {
				Map.Entry<String, Map<Integer, Double>> me2 = it2.next();
				j = Integer.parseInt(me2.getKey());	
				TestDistance[i][j] = getDistance2(CateCenterFactorMap.get(me2.getKey()), me.getValue(),me2.getValue());
			}
		    i++;
		}

		int[] nearestMeans = new int[TestFileMapVsm.size()];
		int[] SecondMeans = new int[TestFileMapVsm.size()];
		for (i = 0; i < TestFileMapVsm.size(); i++) {
			nearestMeans[i] = findNearestMeans(TestDistance, i);
			SecondMeans[i] = findSecondNearestMeans(TestDistance, i);
		}

		i = 0;
		int correct = 0;
		int total   = TestFileMapVsm.size();
		Set<Map.Entry<String, Map<Integer, Double>>> TestFileMapVsmSet2 = TestFileMapVsm.entrySet();
		for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = TestFileMapVsmSet2.iterator(); it.hasNext();) {
			Map.Entry<String, Map<Integer, Double>> me = it.next();
			String Cate = me.getKey().split("_")[0];
			//System.out.println("i: "+i+", Cate: "+Integer.parseInt(Cate)+", nearestMeans: "+nearestMeans[i]);
			if(Integer.parseInt(Cate) == nearestMeans[i])
			{
				correct += 1.0;
			}
			else
			{
				//System.out.println(" "+me.getKey()+"  nearest:"+nearestMeans[i-1]+" , second:"+SecondMeans[i-1]);
			}
			actual.put(me.getKey(), Cate);
			pred.put(me.getKey(), ""+nearestMeans[i]);
			i++;
		}
		System.out.println("Test  Accuracy = "+(double)correct/total*100+"% ("+correct+"/"+total+") (TestResult Classifier)\n");
		return (double)correct/total;
	}

	public static double getDistance2(double beta, Map<Integer, Double> map1,
			Map<Integer, Double> map2) {
		// TODO Auto-generated method stub
		return 1 - beta*computeSim(map1, map2);
	}
	
	public static double computeSim(Map<Integer, Double> testWordTFMap,
			Map<Integer, Double> trainWordTFMap) {
		// TODO Auto-generated method stub
		double mul = 0, testAbs = 0, trainAbs = 0, ret = 0;;
		Set<Map.Entry<Integer, Double>> testWordTFMapSet = testWordTFMap
				.entrySet();
		for (Iterator<Map.Entry<Integer, Double>> it = testWordTFMapSet
				.iterator(); it.hasNext();) {
			Map.Entry<Integer, Double> me = it.next();
			if (trainWordTFMap.containsKey(me.getKey())) {
				mul += me.getValue() * trainWordTFMap.get(me.getKey());
			}
			testAbs += me.getValue() * me.getValue();
		}
		testAbs = Math.sqrt(testAbs);

		Set<Map.Entry<Integer, Double>> trainWordTFMapSet = trainWordTFMap
				.entrySet();
		for (Iterator<Map.Entry<Integer, Double>> it = trainWordTFMapSet
				.iterator(); it.hasNext();) {
			Map.Entry<Integer, Double> me = it.next();
			trainAbs += me.getValue() * me.getValue();
		}
		trainAbs = Math.sqrt(trainAbs);
		ret = mul / (testAbs * trainAbs);
		if(Double.isNaN(ret))
		{
			//System.out.println("computeSim mul: "+mul+" , "+testAbs+" , "+trainAbs);
			ret = 0;
		}
		return ret;
	}

	public static int findNearestMeans(double[][] distance, int m) {
		// TODO Auto-generated method stub
		double minDist = distance[m][0];
		int j = 0;
		for (int i = 0; i < distance[m].length; i++) {
			if (distance[m][i] < minDist) {
				minDist = distance[m][i];
				j = i;
			}
		}
		if(j==0)
		{
			for (int i = 0; i < distance[m].length; i++) {
				System.out.println(i+", "+m+", "+distance[m][i]);
			}
		}
		return j;
	}

	public static int findSecondNearestMeans(double[][] distance, int m) {
		// TODO Auto-generated method stub
		double minDist = distance[m][0];
		int minindex = 0;
		int j = 0;
		for (int i = 0; i < distance[m].length; i++) {
			if (distance[m][i] < minDist) {
				minDist = distance[m][i];
				minindex = i;
			}
		}
		
		minDist = distance[m][0];
		for (int i = 0; i < distance[m].length; i++) {
			if ((i!=minindex)&&(distance[m][i] < minDist)) {
				minDist = distance[m][i];
				j = i;
			}
		}

		return j;
	}

	public static double TestResult(Map<String, Double> CateCenterFactorMap,
			Map<String, Map<Integer, Double>> TestFileMapVsm,
		    Map<String, Map<Integer, Double>> CateCenterMap) throws IOException {
		int i = 0;
		int j = 0;
		double[][] TestDistance = new double[TestFileMapVsm.size()][CateCenterMap.size()+1];
		
		//System.out.println("MultlinearMap.size(): "+MultlinearMap.size());
		for(i=0; i<TestFileMapVsm.size(); i++){
			for(j=0; j<CateCenterMap.size()+1; j++){
				TestDistance[i][j] = 10.0;
			}
		}
		
		i = 0;
		j = 0;
		Set<Map.Entry<String, Map<Integer, Double>>> TestFileMapVsmSet = TestFileMapVsm.entrySet();
		for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = TestFileMapVsmSet.iterator(); it.hasNext();) {
		Map.Entry<String, Map<Integer, Double>> me = it.next();
			j = 0;
			Set<Map.Entry<String, Map<Integer, Double>>> CateCenterMapSet1 = CateCenterMap.entrySet();
			for (Iterator<Map.Entry<String, Map<Integer, Double>>> it2 = CateCenterMapSet1.iterator(); it2.hasNext();) {
				Map.Entry<String, Map<Integer, Double>> me2 = it2.next();
				j = Integer.parseInt(me2.getKey());	
				TestDistance[i][j] = getDistance2(CateCenterFactorMap.get(me2.getKey()), me.getValue(),me2.getValue());
			}
			i++;
		}
		
		int[] nearestMeans = new int[TestFileMapVsm.size()];
		int[] SecondMeans = new int[TestFileMapVsm.size()];
		for (i = 0; i < TestFileMapVsm.size(); i++) {
			nearestMeans[i] = findNearestMeans(TestDistance, i);
			SecondMeans[i] = findSecondNearestMeans(TestDistance, i);
		}
		
		i = 0;
		int correct = 0;
		int total   = TestFileMapVsm.size();
		Set<Map.Entry<String, Map<Integer, Double>>> TestFileMapVsmSet2 = TestFileMapVsm.entrySet();
		for (Iterator<Map.Entry<String, Map<Integer, Double>>> it = TestFileMapVsmSet2.iterator(); it.hasNext();) {
			Map.Entry<String, Map<Integer, Double>> me = it.next();
			String Cate = me.getKey().split("_")[0];
			//System.out.println("i: "+i+", Cate: "+Integer.parseInt(Cate)+", nearestMeans: "+nearestMeans[i]);
			if(Integer.parseInt(Cate) == nearestMeans[i])
			{
				correct += 1.0;
			}
			else
			{
				//System.out.println(" "+me.getKey()+"  nearest:"+nearestMeans[i]+" , second:"+SecondMeans[i]);
			}
			i++;
		}
		System.out.println("Test  Accuracy = "+(double)correct/total*100+"% ("+correct+"/"+total+") (TestResult Classifier)\n");
		return (double)correct/total;
	}

}
