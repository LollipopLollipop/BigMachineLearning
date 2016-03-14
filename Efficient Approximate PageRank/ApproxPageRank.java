import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;


public class ApproxPageRank {
	
	public static void main(String[] args) {
		//long sTime = System.currentTimeMillis();
		if(args.length<4){
			System.err.println("INSUFFICIENT INPUT PARAMETERS");
			System.exit(1);
		}
		String inputPath = args[0];
		String seed = args[1];
		double alpha = Double.parseDouble(args[2]);
		double epsilon = Double.parseDouble(args[3]);
		
		HashMap<String, Double> p = new HashMap<String, Double>();
		HashMap<String, Double> r = new HashMap<String, Double>();
		HashMap<String, Integer> d = new HashMap<String, Integer>();
		HashMap<String, HashSet<String>> neighborsMap = new HashMap<String, HashSet<String>>();
		r.put(seed,1.0);
		int prevRsize = 0;
		double maxRUbyDU = epsilon;
		//boolean firstScan = true;
		while(maxRUbyDU>=epsilon && r.size()>prevRsize){
			//System.out.println("scan starts");
			//System.out.println("max RU/DU|"+maxRUbyDU);
			maxRUbyDU = 0;
			prevRsize = r.size();
			//firstScan = false;
			try {
				BufferedReader br = new BufferedReader(new FileReader(inputPath));
				String thisLine = null;
				while ((thisLine = br.readLine()) != null) {
					int end = thisLine.indexOf("\t");
					String u = thisLine.substring(0, end);
					if(r.containsKey(u)){
						int du = 0;
						HashSet<String> neighbors = new HashSet<String>();
						try{
							du = d.get(u);
							neighbors = neighborsMap.get(u);
						}catch(NullPointerException e){
							int start = end;
							while((end = thisLine.indexOf("\t",start+1))>0){
								String t = thisLine.substring(start+1, end);
								//System.out.println(t);
								neighbors.add(t);
								start = end;
							}
							//System.out.println(thisLine.substring(start+1));
							neighbors.add(thisLine.substring(start+1));
							du = neighbors.size();
						}
						
						//neighborsMap.put(u, neighbors);
						double ru = r.get(u);
						//System.out.println(du);
						double curRUbyDU = ru/du;
						//System.out.println(curRUbyDU);
						maxRUbyDU = Math.max(maxRUbyDU, curRUbyDU);
						if(curRUbyDU >= epsilon){
							try{
								p.put(u,p.get(u)+alpha*ru);
								
							}catch (NullPointerException e){
								//System.out.println("null");
								p.put(u,alpha*ru);
								d.put(u,du);
								neighborsMap.put(u, neighbors);
							}
							r.put(u, (1-alpha)*ru/2);
							
							for(String v:neighbors){
								try{
									r.put(v,r.get(v)+(1-alpha)*ru/(2*du));
									
								}catch (NullPointerException e){
									r.put(v,(1-alpha)*ru/(2*du));
								}
							}
						}
					}	
				}
//				System.out.println("p size|"+p.size());
//				System.out.println("d size|"+p.size());
//				System.out.println("neighborsMap size|"+neighborsMap.size());
//				System.out.println("r size|"+r.size());
			}catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		
		
		
		//Building a low-conductance subgraph
		List<Map.Entry<String, Double>> pList = 
				new LinkedList<Map.Entry<String, Double>>(p.entrySet());
		// Sort list with comparator, to compare the Map values
		Collections.sort(pList, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1,
                                           Map.Entry<String, Double> o2) {
				//return (o1.getValue()).compareTo(o2.getValue());
				if(o1.getValue()>=o2.getValue())
					return -1;
				else
					return 1;
			}
		});
		
		Set<String> S = new HashSet<String>();
		//HashMap<String, String> boundary = new HashMap<String, String>();
		S.add(seed);
		int volS = d.get(seed);
		int boundS = volS;
		Set<String> prevS = new HashSet<String>(S);
		double prevConductance = (double)boundS/volS;
		for(Map.Entry<String, Double> v: pList){
			//System.out.println(v.getKey()+"|"+v.getValue());
			if(!v.getKey().equals(seed)){
				S.add(v.getKey());
				volS+=d.get(v.getKey());
				//update bound, adding the edges from v to any node not belong to S +{v}.
				HashSet<String> neighbors = new HashSet<String>(neighborsMap.get(v.getKey()));
				neighbors.removeAll(S);
				boundS += neighbors.size();
				//removing the set of edges that enter v
				int inlinkToV = 0;
				for(String w:S){
					if(neighborsMap.get(w).contains(v.getKey()))
						inlinkToV++;
				}
				boundS -= inlinkToV;
				double conductance = (double)boundS/volS;
				if(conductance<prevConductance){
					prevS = new HashSet<String>(S);
					prevConductance = conductance;
				}
			}
		}
		
		try {
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
			//BufferedWriter bw = new BufferedWriter(new FileWriter("v2_5.output"));
			for(String v:prevS){
				StringBuilder sb = new StringBuilder();
				sb.append(v);
				sb.append("\t");
				sb.append(p.get(v));
				sb.append("\n");
				bw.write(sb.toString());
			}
			bw.close();
		}catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//long eTime = System.currentTimeMillis();
		//System.out.println("running time: "+(eTime-sTime)/1000);
		// export to GDF format graph
//		try {
//			BufferedWriter bw = new BufferedWriter(new FileWriter("q4_subgraph.gdf"));
//			bw.write("nodedef>name VARCHAR, label VARCHAR, width DOUBLE\n");
//			for(String v:prevS){
//				StringBuilder sb = new StringBuilder();
//				sb.append(v);
//				sb.append(",");
//				sb.append(v);
//				sb.append(",");
//				double size = Math.max(1, Math.log(p.get(v)/epsilon));
//				sb.append(size);
//				sb.append("\n");
//				bw.write(sb.toString());
//			}
//			bw.write("edgedef>node1 VARCHAR,node2 VARCHAR\n");
//			for(String u:prevS){
//				HashSet<String> neighbors = neighborsMap.get(u);
//				System.out.println("orig neighbors size"+neighbors.size());
//				neighbors.retainAll(prevS);
//				System.out.println("after retainAll"+neighbors.size());
//				for(String v:neighbors){
//					StringBuilder sb = new StringBuilder();
//					sb.append(u);
//					sb.append(",");
//					sb.append(v);
//					sb.append("\n");
//					bw.write(sb.toString());
//				}
//				
//			}
//			bw.close();
//		}catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		
	}
	
	
	

}
