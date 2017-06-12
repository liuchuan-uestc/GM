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

public class SVM{
	static double TrainCenterPoint = 0;

	public static double SVMMain(String strDir, Map<String, Map<Integer, Double>> TrainFileMapVsm, 
								           Map<String, Map<Integer, Double>> TestFileMapVsm,
								           Map<String,String> actual, Map<String,String> pred, int i, int j) throws IOException {
		System.out.println("SVM start...i = "+i+", j = "+j);
		String SVM_train_argv = "-s 0 -t 2 -c 128 -g 0.1 -n 0.1 VsmTFIDFMapTrainSample"+j;
		String SVM_test_argv  = "-m 1 VsmTFIDFMapTestSample"+j+" VsmTFIDFMapTrainSample"+j+".model "+"output"+j;
	
		String[] train_argv_Split = SVM_train_argv.split(" ");
		String[] test_argv_Split  = SVM_test_argv.split(" ");
		svm_train   SVM_train = new svm_train();
				
		svm_predict SVM_test  = new svm_predict();
		svm_model model = SVM_train.run(train_argv_Split, TrainFileMapVsm);

		double accuracy = SVM_test.run(test_argv_Split, model, TestFileMapVsm, actual, pred);
		return accuracy;
	}

}
