import java.util.HashMap;


public class TrainingInstance {
	private String y = null;
	private HashMap<Integer, Integer> X = new HashMap<Integer, Integer>();
	public TrainingInstance(String y, HashMap<Integer, Integer> X){
		this.y = y;
		this.X = X;
	}
	public String getY(){
		return this.y;
	}
	public HashMap<Integer, Integer> getX(){
		return this.X;
	}
}
