import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;


public class LR {

	public static void main(String[] args) {
		//long startTime = System.currentTimeMillis();
		//params parsing
		int vocSize = Integer.parseInt(args[0]);
		
		double initLearnRate = Double.parseDouble(args[1]);
		
		double mu = Double.parseDouble(args[2]);
		
		int maxIter = Integer.parseInt(args[3]);
		
		int trainSize = Integer.parseInt(args[4]);
		
		String testDataPath = args[5];
		
		//HashMaps to hold training data and classifier params
		ArrayList<TrainingInstance> trainingData = new ArrayList<TrainingInstance>();
		//HashMap<String, HashMap<Integer, Integer>> A = new HashMap<String, HashMap<Integer, Integer>>();
		HashMap<String, HashMap<Integer, Double>> B = new HashMap<String, HashMap<Integer, Double>>();
		//14 binary classifier target
		List<String> labelsCollection = Arrays.asList("nl","el","ru","sl","pl","ca","fr","tr","hu","de","hr","es","ga","pt");
		//List<String> labelsCollection = Arrays.asList("fr");
		for(String l:labelsCollection){
			//A.put(l, new HashMap<Integer, Integer>());
			B.put(l, new HashMap<Integer, Double>());	
		}
		
		try {
			//read in training data
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			//BufferedReader br = new BufferedReader(new FileReader(args[7]));
			String thisLine = null;
			Random rand = new Random();
			while ((thisLine = br.readLine()) != null) {
				float f = rand.nextFloat();
				if(f>0.1)
					continue;
				int tabIdx = thisLine.indexOf("\t");
				
	            StringTokenizer labels = new StringTokenizer(thisLine.substring(0, tabIdx), ",");
	            StringTokenizer words = new StringTokenizer(thisLine.substring(tabIdx+1));
	            //Map from word to their occurrence freq
	            HashMap<Integer, Integer> wordsFreqMap = new HashMap<Integer, Integer>();
	            while (words.hasMoreTokens()) {
					String word = words.nextToken();
					Integer wordCode = hashing(word,vocSize);
					try{
						int freq = wordsFreqMap.get(wordCode);
						wordsFreqMap.put(wordCode, freq+1);
					}catch(NullPointerException e){
						wordsFreqMap.put(wordCode, 1);
					}
				
	            }
	            
	            while (labels.hasMoreTokens()){
	            	String label = labels.nextToken();
	            	//compose <y_train, X_train> pair
	            	TrainingInstance ins = new TrainingInstance(label, wordsFreqMap);
	            	trainingData.add(ins);
	            }
			}
	            	
			br.close();

		}catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//System.out.println(trainingData.size());
		trainSize = trainingData.size()/maxIter;		
		//training 14 classifier iteratively
		for(String classLabel: B.keySet()){
			//training routine for each binary class
			HashMap<Integer, Integer> APerClass = new HashMap<Integer, Integer>();
			HashMap<Integer, Double> BPerClass = B.get(classLabel);
			int k=0;
			int scanCount=1;
			double lambda = initLearnRate;
			for(TrainingInstance ins: trainingData){
				String yTrain = ins.getY();
				k++;
				//adjust learning rate
				if((k/trainSize)>=scanCount){
					scanCount++;
					lambda = initLearnRate/(scanCount*scanCount);
				}
				HashMap<Integer,Integer> wordFreqVector = ins.getX();
				for(int j=0; j<vocSize; j++){
//					int freq;
//					try{
//						freq = wordFreqVector.get(j);
//					}catch(NullPointerException e){
//						continue;
//					}
					if(wordFreqVector.containsKey(j)){
						int freq = wordFreqVector.get(j);
						if(!BPerClass.containsKey(j)){
							BPerClass.put(j, 0.0);
							APerClass.put(j, 0);
							
						}else{
							double regBj = regUpdate(BPerClass.get(j),lambda,mu,k,APerClass.get(j));
			    			BPerClass.put(j,regBj);
						}
		//    				double Bj=0.0;
		//    				int Aj=0;
		//    				try{
		//    					Bj = BPerClass.get(j);
		//    					Aj = APerClass.get(j);
		//    				}catch(NullPointerException e){
		//    					BPerClass.put(j, 0.0);
		//    					APerClass.put(j, 0);
		//					}
		    				
		    			
		    			double prob = calcProb(wordFreqVector,BPerClass);
						if(classLabel.equals(yTrain)){
							double updatedBj = BPerClass.get(j)+lambda*(1-prob)*freq;
							BPerClass.put(j, updatedBj);
						}
						else{
							double updatedBj = BPerClass.get(j)+lambda*(0-prob)*freq;
							BPerClass.put(j, updatedBj);
						}
		    			APerClass.put(j,k);
					}
		    	}
		    }
			
			
			for(int j:BPerClass.keySet()){
				double regBj = regUpdate(BPerClass.get(j),lambda,mu,k,APerClass.get(j));
    			BPerClass.put(j,regBj);
			}
			
		}
		
		//testing
		try{
			BufferedReader br = new BufferedReader(new FileReader(testDataPath));
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
            //BufferedWriter bw = new BufferedWriter(new FileWriter(args[6]));
			String thisLine = null;
			while ((thisLine = br.readLine()) != null) {
				int tabIdx = thisLine.indexOf("\t");
				StringTokenizer words = new StringTokenizer(thisLine.substring(tabIdx+1));
				
				//Map from word to their occurrence freq
	            HashMap<Integer, Integer> wordsFreqMap = new HashMap<Integer, Integer>();
	            while (words.hasMoreTokens()) {
					String word = words.nextToken();
					Integer wordCode = hashing(word,vocSize);
					if(wordsFreqMap.containsKey(wordCode)){
						wordsFreqMap.put(wordCode, wordsFreqMap.get(wordCode)+1);
					}else{
						wordsFreqMap.put(wordCode, 1);
					}
	            }
	            
	            
	            StringBuilder sb = new StringBuilder(); 
	            for(String classLabel:B.keySet()){
	            	HashMap<Integer, Double> BPerClass = B.get(classLabel);
	            	//System.out.println("BPerClass key size"+BPerClass.keySet().size());
	            	double prob = calcProb(wordsFreqMap,BPerClass);
	            	sb.append(classLabel+"\t"+Math.round(prob*100000000)/100000000.0+",");
	            }
	            sb.setLength(sb.length()-1);
	            sb.append("\n");
	            bw.write(sb.toString());
				//bw.flush();
			}
			br.close();
			bw.close();
			//System.out.println("running time "+(System.currentTimeMillis()-startTime)/1000);
		}catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
		
	
		
	
	
	private static double regUpdate(double Bj, double lambda, double mu,
			int k, int Aj) {
//		if(Double.isNaN(Bj)){
//			System.err.println("given Bj is NaN");
//		}
		double newBj = Bj*Math.pow(1-2*lambda*mu,k-Aj);
//		if(Double.isNaN(newBj)){
//			System.err.println("new Bj is NaN");
//		}
		return newBj;
	}





	private static Double calcProb(HashMap<Integer, Integer> wordsFeatureVector, HashMap<Integer, Double> BPerClass){
		//return 0.0;
		Double sum = 0.0;
		for(Integer i:wordsFeatureVector.keySet()){
//			if(BPerClass.containsKey(i)&&(BPerClass.get(i)!=0)){
//				
//			}
			try{
				sum+=wordsFeatureVector.get(i)*BPerClass.get(i);
				if(sum>5)
					return 1.0;
			}catch(NullPointerException e){
				
			}
		}
		Double exp = Math.exp(sum);
		Double prob = exp/(1+exp);
		return prob;
//		if(exp.equals(Double.POSITIVE_INFINITY))
//			return 1.0;
//		else{
//			Double prob = exp/(1+exp);
//			return prob;
//		}
	}
	
	
	private static int hashing(String word, int dicSize){
		//N is the dictionary size
		int id = word.hashCode() % dicSize;
		if (id<0) id+= dicSize;
		return id;
	}

}
