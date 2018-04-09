package org.tucci.psl.coll.hybrid;

import java.io.IOException;
import java.text.DecimalFormat
import java.util.ArrayList
import java.util.Map;

import org.linqs.psl.application.inference.MPEInference;
import org.linqs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import org.linqs.psl.config.ConfigBundle;
import org.linqs.psl.config.ConfigManager;
import org.linqs.psl.database.Database;
import org.linqs.psl.database.DatabasePopulator;
import org.linqs.psl.database.DataStore;
import org.linqs.psl.database.Partition;
import org.linqs.psl.database.Queries;
import org.linqs.psl.database.ReadOnlyDatabase;
import org.linqs.psl.model.term.ConstantType;
import org.linqs.psl.database.loading.Inserter;
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver;
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import org.linqs.psl.database.rdbms.RDBMSDataStore;
import org.linqs.psl.groovy.PSLModel;
import org.linqs.psl.model.atom.Atom;
import org.linqs.psl.model.atom.GroundAtom;
import org.linqs.psl.model.atom.RandomVariableAtom;
import org.linqs.psl.model.predicate.Predicate;
import org.linqs.psl.model.predicate.StandardPredicate;
import org.linqs.psl.model.term.ConstantType;
import org.linqs.psl.utils.dataloading.InserterUtils;
import org.linqs.psl.utils.evaluation.printing.AtomPrintStream;
import org.linqs.psl.utils.evaluation.printing.DefaultAtomPrintStream;
import org.linqs.psl.utils.evaluation.statistics.ContinuousPredictionComparator;
import org.linqs.psl.utils.evaluation.statistics.DiscretePredictionComparator;
import org.linqs.psl.utils.evaluation.statistics.DiscretePredictionStatistics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import groovy.time.TimeCategory;
import java.nio.file.Paths;


/**
 * Model 1 for Hybrid Data : 
 * Collective Classification for Hybrid Data
 * This model can be used in order to generate results for Visitor Stitching
 * In order to apply given model to propreitary data, generate psl rules by going through 
 * your data and rename variables if required.
 **/
public class HybridCCM1 {
    private static final String PARTITION_TRAIN_OBSERVATIONS = "trainObservations";
	private static final String PARTITION_TRAIN_TARGETS = "trainTargets";
	private static final String PARTITION_TRAIN_TRUTH = "trainTruth";

	private static final String PARTITION_TEST_OBSERVATIONS = "testObservations";
    private static final String PARTITION_TEST_TARGETS = "testTargets";
	private static final String PARTITION_TEST_TRUTH = "testTruth";

	private static final String PARTITION_EVAL_OBSERVATIONS = "evalObservations";
    private static final String PARTITION_EVAL_TARGETS = "evalTargets";
	private static final String PARTITION_EVAL_TRUTH = "evalTruth";

	private Logger log;
	private DataStore ds;
	private PSLConfig config;
	private PSLModel model;

	private Database trainDB;
	private Database trainAnswerDB;
	private Database testDB;
	private Database evalDB;

	private static weightExtensionFileName;

	/**
	 * Class containing options for configuring the PSL program
	 */
	// @groovy.transform.InheritConstructors
	private class PSLConfig {
		//////////////////////////// Configuration ////////////////////////////
		ConfigManager cm = ConfigManager.getManager()
		public ConfigBundle cb = cm.getBundle("Baseline")
		
		public String experimentName;
		public String dbPath;
		public String dataPath;
		public String outputPath;
		public Integer trainIndex;
		public Integer evalIndex;
		public Integer testIndex;
		public int removeRule;
		public Boolean createNewDatastore;
		////////////////////////  Model description ///////////////////////////
		public double initialWeight = 1;
		public boolean sq = true;

		public def preds;
		public def pred_to_filepred;

		public PSLConfig(ConfigBundle cb) {
			this.cb = cb;

			this.experimentName = cb.getString('experiment.name', 'default');
			this.dbPath = cb.getString('experiment.dbpath', '/tmp/hybrid_collective');
			this.dataPath = cb.getString('experiment.data.path', '/data');
			this.outputPath = cb.getString('experiment.output.outputdir', Paths.get('output', this.experimentName).toString());
			this.trainIndex = cb.getInteger('experiment.index.train', 0);
			this.evalIndex = cb.getInteger('experiment.index.eval', 0);
			this.testIndex = cb.getInteger('experiment.index.test', 0);
			this.removeRule = cb.getInteger('model.rules.remove', 0);

			this.createNewDatastore = true;

			// this.preds = [Block, Active, Device, GeoLoc, IP, Browser, URL1, SimURL, SimIP, VeryClose, \
			// 		DevActive, DevBrw, UserSpaceTime, Far];
			// this.pred_to_filepred = [Block:'BLOCK', Active:'ACTIVE', Device:'DV', GeoLoc:'LOC', IP:'IP', \
			// 					Browser:'BR', URL1:'URL', SimURL:'URLSIM', SimIP:'IPSIM', VeryClose:'VeryClose', \
			// 					DevActive:'DEVACTIVE', DevBrw:'DEVBRW', UserSpaceTime:'SPACETIME', Far:'Far'];
		}
	}
	
	/**
	 * Defines the logical predicates used by this program
	 */
	private void definePredicates() {
		//this.model.add predicate: "UID", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]     //CookieID, UserLoginID
		this.model.add predicate: "Block", types: [ConstantType.UniqueID, ConstantType.UniqueID]   //CookieID, RegionID
		this.model.add predicate: "Active", types: [ConstantType.UniqueID, ConstantType.UniqueID]  //CookieID, DateTimeID
		this.model.add predicate: "Device", types: [ConstantType.UniqueID, ConstantType.UniqueID]  //CookieID, DeviceID
		this.model.add predicate: "GeoLoc", types: [ConstantType.UniqueID, ConstantType.UniqueID]     //CookieID, LocID
		this.model.add predicate: "IP", types: [ConstantType.UniqueID, ConstantType.UniqueID]      //CookieID, IPID
		this.model.add predicate: "Browser", types: [ConstantType.UniqueID, ConstantType.UniqueID] //CookieID, BrowserID
		this.model.add predicate: "URL1", types: [ConstantType.UniqueID, ConstantType.UniqueID]    //CookieID, URLID
		this.model.add predicate: "VeryClose", types: [ConstantType.UniqueID, ConstantType.UniqueID]  //LocID, LocID
		this.model.add predicate: "SimIP", types: [ConstantType.UniqueID, ConstantType.UniqueID]  //IPID, IPID
		this.model.add predicate: "SimURL", types: [ConstantType.UniqueID, ConstantType.UniqueID]  //URLID, URLID
		this.model.add predicate: "DevActive", types: [ConstantType.UniqueID, ConstantType.UniqueID, ConstantType.UniqueID] //CookieID, DeviceID, DateTimeID
		this.model.add predicate: "DevBrw", types: [ConstantType.UniqueID, ConstantType.UniqueID, ConstantType.UniqueID]    //CookieID,BrowserID, DateTimeID
		this.model.add predicate: "Far", types: [ConstantType.UniqueID, ConstantType.UniqueID]  //LocID, LocID
		this.model.add predicate: "UserSpaceTime", types: [ConstantType.UniqueID, ConstantType.UniqueID, ConstantType.UniqueID]  //CookieID, LocID, DateTimeID

		// target predicate
		this.model.add predicate: "SameUser", types: [ConstantType.UniqueID, ConstantType.UniqueID]
	}

	/**
	 * Defines the rules used to infer unknown variables in the PSL program
	 */
	private void defineRules() {
		log.info("Defining model rules");
		int rule_to_remove = this.config.removeRule;
		
		if (rule_to_remove != 1) {
			// R4: If the IP Address is same up to 3 octets, it is the same user
			this.model.add(
				rule : ( (A-B) & Block(A, X) & Block(B, X) & IP(A, I1) & IP(B, I2) & SimIP(I1, I2)) >> SameUser(A, B), 
				weight : 0.75
			);
		}
		if (rule_to_remove != 2) {
			// R5: Transitivity Rule
			this.model.add(
				rule : ( (A-B) & (B-C) & (A-C) & Block(A, X) &  Block(B, X) & Block(C, X) & SameUser(A, B) & SameUser(B, C)) >> SameUser(A, C), 
				weight : 1
			);
		}
		if (rule_to_remove != 3) {
			// R6: If two users are using the same device, but different browsers, it is likely not the same user
			this.model.add(
				rule : ( (A-B) & (B1-B2) & Block(A, X) &  Block(B, X) & DevBrw(A, D, B1) & DevBrw(B, D, B2) ) >> ~SameUser(A, B), 
				weight : 0.75
			);
		}
		if (rule_to_remove != 4) {
			// R8: If users A & B are at very distant locations in the same hour, it is not the same user
			this.model.add(
				rule : ( (A-B) & (S1-S2) & UserSpaceTime(A, S1, T) & UserSpaceTime(B, S2, T) & Far(S1, S2)) >> ~SameUser(A, B), 
				weight : 0.5
			);
		}
		if (rule_to_remove != 5) {
			// R2: 3 cookies active at the same time can't belong to the same user
			this.model.add(
				rule : ( (A-B) & (B-C) & (A-C) & Block(A, X) &  Block(B, X) &  Block(C, X) & Active(A, T) & Active(B, T) & Active(C, T) & SameUser(A, B)) >> ~SameUser(A, C), 
				weight : 0.5
			);
		}
		if (rule_to_remove != 6) {
			// R1: If the GeoLocation is the same, it is the same user
			this.model.add(
				rule : ( (A-B) &  Block(A, X) &  Block(B, X) & GeoLoc(A, G1) & GeoLoc(B, G2) & VeryClose(G1, G2)) >> SameUser(A, B), 
				weight : 1
			);
		}
		if (rule_to_remove != 7) {
			// R7: Visits to the SameURL means it is likely the Same User - distance metric needs to be discussed
			this.model.add(
				rule : ( (A-B) & Block(A, X) & Block(B, X) & URL1(A, U1) & URL1(B, U2) & SimURL(U1, U2)) >> SameUser(A, B),  
				weight : 0.5
			);
		}
		if (rule_to_remove != 8) {
			// R3: A person can't be active on the same type of device at the same time
			this.model.add(
				rule : ( (A-B) & Block(A, X) &  Block(B, X) & DevActive(A, D, T) & DevActive(B, D, T) ) >> ~SameUser(A, B), 
				weight : 0.5
			);
		}
		if (rule_to_remove != 9) {
			// R9: Negative prior
			this.model.add(
				rule: ( (A-B) & Block(A, X) & Block(B, X) ) >> ~SameUser(A, B), 
				weight : 1 
			);
		}
		if (rule_to_remove != 10) {
			// this.model.add(
			// 	rule: SameUser(A, B) - SameUser(B, A) = 0
			// );
		}
	}

	private int getPositionOfPredicate(Predicate p) {
		def preds = [Block, Active, Device, GeoLoc, IP, Browser, URL1, SimURL, SimIP, VeryClose, \
					DevActive, DevBrw, UserSpaceTime, Far];
		int position = -1;
		for(int i = 0; i < preds.size; ++i) {
			if(p.getName().equalsIgnoreCase(preds[i].getName())) {
				position = i;
			}
		}	

		return position;
	}

	private void loadTestData(Partition obsTestPartition, Partition targetsTestPartition, Partition truthTestPartition) {
		def dir = "/home/kapil/databases/newdata/test_data/" + this.config.dataPath + "/";

		def insert = this.ds.getInserter(SameUser, truthTestPartition);
		InserterUtils.loadDelimitedDataTruth(insert, dir + "SameUser" + this.config.testIndex);
		Database testAnswerDB = this.ds.getDatabase(truthTestPartition, [SameUser] as Set);

		// def preds = [Block, Active, Device, GeoLoc, IP, Browser, URL1, SimURL, SimIP, VeryClose, \
		// 			DevActive, DevBrw, UserSpaceTime, Far];
		def preds = [Block];

		// Get evidences for testing
		for (Predicate p : preds)
		{
			def x = p.getName();
			insert = this.ds.getInserter(p, obsTestPartition);
			
			if (x == 'VERYCLOSE' || x == 'URLSIM' || x == 'IPSIM'|| x == 'FAR')
					InserterUtils.loadDelimitedDataTruth(insert, dir + x + this.config.testIndex)
			else
					InserterUtils.loadDelimitedData(insert, dir + x + this.config.testIndex);
		}

		this.testDB = this.ds.getDatabase(targetsTestPartition, this.config.preds as Set, obsTestPartition);
		populateSameUser(this.testDB);
	}

	private void loadTrainData(Partition obsTrainPartition, Partition targetsTrainPartition, Partition truthTrainPartition) {
		def dir = "/home/kapil/databases/newdata/train_data/" + this.config.dataPath + "/";

		def insert = this.ds.getInserter(SameUser, truthTrainPartition);
		InserterUtils.loadDelimitedDataTruth(insert, dir + "SameUser" + this.config.trainIndex);
		this.trainAnswerDB = this.ds.getDatabase(truthTrainPartition, [SameUser] as Set);

		// def preds = [Block, Active, Device, GeoLoc, IP, Browser, URL1, SimURL, SimIP, VeryClose, \
		// 			DevActive, DevBrw, UserSpaceTime, Far];
		def preds = [Block];

		// Get evidences for training
		for (Predicate p : preds)
		{
			insert = this.ds.getInserter(p, obsTrainPartition);
			def x = p.getName();
			
			if (x == 'VERYCLOSE' || x == 'URLSIM' || x == 'IPSIM'|| x == 'FAR')
					InserterUtils.loadDelimitedDataTruth(insert, dir + x + this.config.trainIndex)
			else
					InserterUtils.loadDelimitedData(insert, dir + x + this.config.trainIndex);
		}

		this.trainDB = this.ds.getDatabase(targetsTrainPartition, this.config.preds as Set, obsTrainPartition);
		populateSameUser(this.trainDB);
	}

	private void loadEvalData(Partition obsEvalPartition, Partition targetsEvalPartition, Partition truthEvalPartition) {
		def dir = "/home/kapil/databases/newdata/eval_data/" + this.config.dataPath + "/";

		def insert = this.ds.getInserter(SameUser, truthEvalPartition);
		InserterUtils.loadDelimitedDataTruth(insert, dir + "SameUser" + this.config.evalIndex);
		Database evalAnswerDB = this.ds.getDatabase(truthEvalPartition, [SameUser] as Set);

		// def preds = [Block, Active, Device, GeoLoc, IP, Browser, URL1, SimURL, SimIP, VeryClose, \
		// 			DevActive, DevBrw, UserSpaceTime, Far];
		def preds = [Block];

		// Get evidences for evaluation
		for (Predicate p : preds)
		{
			insert = this.ds.getInserter(p, obsEvalPartition);
			def x = p.getName();
			
			if (x == 'VERYCLOSE' || x == 'URLSIM' || x == 'IPSIM'|| x == 'FAR')
					InserterUtils.loadDelimitedDataTruth(insert, dir + x + this.config.trainIndex);
			else
					InserterUtils.loadDelimitedData(insert, dir + x + this.config.trainIndex);
		}

		this.evalDB = this.ds.getDatabase(targetsEvalPartition, this.config.preds as Set, obsEvalPartition);
		populateSameUser(this.evalDB);
	}

	/*
		Original weights
		2.030054587 R4
		1.000000209 R5
		0.846936456 R9
		0.6917437617    R6
		0.4693048537    R8
		0.211666242 R2
		0.1327880217    R1
		0.04744075967   R7
		0.02727796567   R3
	*/

	private void learnWeights() {
		long weightLearningStartTime = System.currentTimeMillis();
		println "LEARNING WEIGHTS...";

		MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(this.model, this.trainDB, this.trainAnswerDB, this.config.cb);
		weightLearning.learn();
		weightLearning.close();

		println "LEARNING WEIGHTS DONE";
		long estimatedWeightLearningTime = System.currentTimeMillis() - weightLearningStartTime;
		println "Weight Learning took "+estimatedWeightLearningTime/1000+" seconds";

		this.trainDB.close();
		this.trainAnswerDB.close();
	}

	private void evalInference(Partition obsEvalPartition, Partition targetsEvalPartition, Partition truthEvalPartition) {
		insert = this.ds.getInserter(SameUser, truthEvalPartition);
		InserterUtils.loadDelimitedDataTruth(insert, dir + "SameUser" + this,config.evalIndex);
		Database evalAnswerDB = this.ds.getDatabase(truthEvalPartition, [SameUser] as Set);

		// Get evidences for evaluation
		for (Predicate p : this.config.preds)
		{
			insert = this.ds.getInserter(p, obsEvalPartition)
			def x = p.getName();
			
			if (x == 'VeryClose' || x == 'URLSIM' || x == 'IPSIM' || x == 'Far')
					InserterUtils.loadDelimitedDataTruth(insert, dir + x + this.config.evalIndex)
			else
					InserterUtils.loadDelimitedData(insert, dir + x + this.config.evalIndex);
		}
		this.evalDB = this.ds.getDatabase(evalPredictions, preds as Set, obsEvalPartition);
		populateSameUser(this.evalDB);

		long evalStartTime = System.currentTimeMillis();
		println "EVAL INFERENCE...";

		MPEInference inference = new MPEInference(this.model, this.evalDB, this.config.cb);
		inference.mpeInference();
		inference.close();

		println "EVAL INFERENCE DONE";
		long estimatedEvalTime = System.currentTimeMillis() - evalStartTime;
		println "Eval took "+estimatedEvalTime/1000+" seconds";

		BufferedReader reader = new BufferedReader(new FileReader(dir + "SameUser"+this.config.evalIndex));
		Map<String, Double> answer_map = new HashMap<String, Double>();

		String line = "";
		try {
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.trim().split("\t");
				String id1 = tokens[0];
				String id2 = tokens[1];
				double y = Double.parseDouble(tokens[2]);
				String pair = id1 + "," + id2;
				answer_map.put(pair, y);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		println "answer_map just read from the SameUser file and it contains " + answer_map.size() + " elements"

		Map<String, Double> sameuseratoms = new HashMap<String, Double>();
		int countgroundatoms = 0
		def resfile = new File(defaultPath + File.separator + runid + '.res')
		for (GroundAtom res : Queries.getAllAtoms(this.evalDB, SameUser)) {
			int temp1 = res.getArguments()[0].getID()
			int temp2 = res.getArguments()[1].getID()
			String pair = temp1 + "," + temp2;
			double y_est = res.getValue();
			sameuseratoms.put(pair,y_est)
			countgroundatoms++
		}

		ArrayList<Label> eval_list = new ArrayList<Label>();

		for (e in answer_map) {
			def ans = e.value
			if (sameuseratoms.containsKey(e.key)){
				def est = sameuseratoms.get(e.key)
				eval_list.add(new Label(est, ans))
				resfile << e.key + '\t' + est + '\t' + ans + '\n'
			}
			else{
				eval_list.add(new Label(0,e.value))
				resfile << e.key + '\t' + 0 + '\t' + ans + '\n'
			}
		}

		println "In Eval, getAllAtoms returned " + countgroundatoms + " SameUser ground atoms of which "+ eval_list.size() +" were added to the eval_list"

		this.evalDB.close();

		double theta = getTheta(eval_list, answer_map);
		resfile << "Optimal theta: "+theta + '\n'
	}

	private void testInference(Partition obsTestPartition, Partition targetsTestPartition, Partition truthTestPartition) {
		insert = this.ds.getInserter(SameUser, truthTestPartition);
		InserterUtils.loadDelimitedDataTruth(insert, dir + "SameUser" + this.config.testIndex);
		Database testAnswerDB = this.ds.getDatabase(truthTestPartition, [SameUser] as Set);

		// Get evidences for testing
		for (Predicate p : preds)
		{
			insert = this.ds.getInserter(p, obsTestPartition)
			x = this.config.pred_to_filepred[p.altName]
			
			if (x == 'VeryClose' || x == 'URLSIM' || x == 'IPSIM')
					InserterUtils.loadDelimitedDataTruth(insert, dir + x + tstidx)
			else
					InserterUtils.loadDelimitedData(insert, dir + x + tstidx);
		}
		this.testDB = this.ds.getDatabase(targetsTestPartition, preds as Set, obsTestPartition);
		populateSameUser(this.testDB);

		long testStartTime = System.currentTimeMillis();
		println "TEST INFERENCE...";

		MPEInference testInference = new MPEInference(this.model, this.testDB, this.config);
		testInference.mpeInference();
		testInference.close();

		println "TEST INFERENCE DONE";
		long estimatedTestTime = System.currentTimeMillis() - testStartTime;
		println "Test took "+estimatedTestTime/1000+" seconds";

		reader = new BufferedReader(new FileReader(dir + "SameUser"+ this.config.testIndex))
		answer_map = new HashMap<String, Double>();

		try {
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.trim().split("\t");
				String id1 = tokens[0];
				String id2 = tokens[1];
				double flag = Double.parseDouble(tokens[2]);
				String pair = id1 + "," + id2;
						
				answer_map.put(pair, flag);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		DecimalFormat formatter = new DecimalFormat("#.####");
		double rmse = 0, test_count = 0;
		for (GroundAtom res : Queries.getAllAtoms(this.testDB, SameUser)) {
			String temp1 = res.getArguments()[0].value//toString()
			String temp2 = res.getArguments()[1].value//toString()
			String pair = temp1 + "," + temp2;
			
			//double value = res.getValue();
			//rmse += Math.pow(value - answer_map.get(pair), 2);
			test_count++;
			
			double y_est = res.getValue();
			
			if (answer_map.containsKey(pair)) {
				estimation_list.add(new Label(y_est, answer_map.get(pair)));
			}
		}

		println "In test, getAllAtoms returned " + test_count + " SameUser ground atoms of which "+estimation_list.size() +" were added to the estimation_list"
		this.testDB.close();

		//String res = getResult(theta, estimation_list);

		//println "Result:\n"+res
	}

	private void writeOutput() {

	}

	public void run(ConfigBundle cb) {
		log.info("Running experiment {}", config.experimentName);

		Partition obsTrainPartition = ds.getPartition(PARTITION_TRAIN_OBSERVATIONS);
		Partition targetsTrainPartition = ds.getPartition(PARTITION_TRAIN_TARGETS);
		Partition truthTrainPartition = ds.getPartition(PARTITION_TRAIN_TRUTH);

        Partition obsTestPartition = ds.getPartition(PARTITION_TEST_OBSERVATIONS);
		Partition targetsTestPartition = ds.getPartition(PARTITION_TEST_TARGETS);
		Partition truthTestPartition = ds.getPartition(PARTITION_TEST_TRUTH);

		Partition obsEvalPartition = ds.getPartition(PARTITION_EVAL_OBSERVATIONS);
		Partition targetsEvalPartition = ds.getPartition(PARTITION_EVAL_TARGETS);
		Partition truthEvalPartition = ds.getPartition(PARTITION_EVAL_TRUTH);

		definePredicates();
		defineRules();

		loadTrainData(obsTrainPartition, targetsTrainPartition, truthTrainPartition);
        loadTestData(obsTestPartition, targetsTestPartition, truthTestPartition);
		loadEvalData(obsEvalPartition, targetsEvalPartition, truthEvalPartition);

		learnWeights();
		// testInference(obsTestPartition, targetsTestPartition, truthTestPartition);
		// evalInference(obsEvalPartition, targetsEvalPartition, truthEvalPartition);
		this.testDB.close();
		this.evalDB.close();
		ds.close();
	}

	
	class Label {
		double estimation;
		double answer;

		public Label(double _estimation, double _answer) {
			this.estimation = _estimation;
			this.answer = _answer;
		}
	}


	void populateSameUser(Database database) {
		Set<GroundAtom> pairs = Queries.getAllAtoms(database, Block);
		for (GroundAtom atom1 : pairs) {
			ConstantType gterm1 = atom1.getArguments()[0];
			ConstantType block1 = atom1.getArguments()[1];
			for (GroundAtom atom2 : pairs) {
				ConstantType gterm2 = atom2.getArguments()[0];
				ConstantType block2 = atom2.getArguments()[1];
				if (block1 == block2 && gterm1 != gterm2){
					((RandomVariableAtom) database.getAtom(SameUser, gterm1, gterm2)).commitToDB();
				}
			}
		}
	}

	double getTheta(ArrayList<Label> estimation_list, HashMap<String, Double> answer_map) {
		
		double optTheta = 0.0, maxScr = 0.0;
		DecimalFormat formatter = new DecimalFormat("#.####");
		
		double tp, fp, tn, fn, acc, pre, rec, fpr, tnr, fscr;
		
		Collections.sort(estimation_list, new Comparator<Label>() {
			public int compare(Label left, Label right) {
				return Double.compare(left.estimation, right.estimation);
			}
		});
		
		double pcnt = 0, ncnt = 0;

		for (e in answer_map) {
		//#for (int i=0; i < answer_map.size(); i++) {
			//#Label l = estimation_list.get(i);
			if (e.value == 0) {
				ncnt++; 
			} else {
				pcnt++;
			}
			
			// println l.estimation + "," + l.answer;
		}
		
		// for (double theta = 0; theta < 1.0; theta += 0.01) {
		double theta = 0, acc_pcnt = 0, acc_ncnt = 0;
		for (int i=0; i<estimation_list.size(); i++) {
			
			Label l = estimation_list.get(i);
			theta = l.estimation;
			
			if (l.answer > 0) {
				acc_pcnt++;
			} else {
				acc_ncnt++;
			}
			
			// double[] subResult = GerContingencyTable(theta, estValSet, ansValSet);
			tp = pcnt - acc_pcnt; //subResult[0];
			fp = ncnt - acc_ncnt; //subResult[1];
			tn = acc_ncnt;//subResult[2];
			fn = acc_pcnt; //subResult[3];

			acc = (tp + tn) / (tp + fp + tn + fn);
			pre = (tp + fp) == 0 ? 0 : tp / (tp + fp);
			rec = (tp + fn) == 0 ? 0 : tp / (tp + fn);
			fpr = (tn + fp) == 0 ? 0 : fp / (tn + fp);
			tnr = (tn + fp) == 0 ? 0 : tn / (tn + fp);
			fscr = 2 * pre * rec / (pre + rec);
					
			if (pre + rec == 0) break;
			
			if (maxScr == 0.0 || maxScr <= fscr) {
				maxScr = fscr;
				optTheta = theta;
			}
			//println theta + ", " + pre + ", " + rec + ", " + fscr
		}
		
		println "Optimized theta is " + optTheta + " where scr is " + formatter.format(maxScr);
		// optTheta = Math.round(optTheta * 10000.0) / 10000.0;
		
		return optTheta;
	}

	/**
	 * Populates the ConfigBundle for this PSL program using arguments specified on
	 * the command line
	 * @param args - Command line arguments supplied during program invocation
	 * @return ConfigBundle with the appropriate properties set
	 */
	public static ConfigBundle populateConfigBundle(String[] args) {
		ConfigBundle cb = ConfigManager.getManager().getBundle("hybridcc");
		if (args.length > 0) {
			cb.setProperty('experiment.name', args[0]);
			cb.setProperty('experiment.dbpath', args[1]);
			cb.setProperty('experiment.data.path', args[2]);
			cb.setProperty('experiment.output.outputDir', args[3]);
			cb.setProperty('experiment.index.train', args[4]);
			cb.setProperty('experiment.index.eval', args[5]);
			cb.setProperty('experiment.index.test', args[6]);
			cb.setProperty('model.rules.remove', args[7]);
		}
		return cb;
	}

	String printResult(double opt_theta, ArrayList<Label> estimation_list, HashMap<String, Double> answer_map) {
		
		println "Opt theta is " + opt_theta
		//println "estimation_list size is " + estimation_list.size()
		
		DecimalFormat formatter = new DecimalFormat("#.####");
		
		double acc, pre, rec, fpr, tnr, fscr;
		
		Collections.sort(estimation_list, new Comparator<Label>() {
			public int compare(Label left, Label right) {
				return Double.compare(left.estimation, right.estimation);
			}
		});

		double pcnt = 0, ncnt = 0, fp = 0, tp = 0, fn = 0 ,tn = 0;
		
		for (e in answer_map) {
			if (e.value == 0) {
				ncnt++; 
			} else {
				pcnt++;
			}
		}        

		for (int i=0; i<estimation_list.size() ; i++) {
			Label l = estimation_list.get(i);
			if (l.answer == 1.0 && l.estimation >= opt_theta) tp++;
			else if (l.answer == 1.0 && l.estimation < opt_theta) fn++;
			else if (l.answer == 0.0 && l.estimation >= opt_theta) fp++;
			else if (l.answer == 0.0 && l.estimation < opt_theta) tn++;
			else println "one fell through the net: l.answer = "+l.answer + ", l.estimation = "+l.estimation
		}

		acc = (tp + tn) / (tp + fp + tn + fn);
		pre = (tp + fp) == 0 ? 0 : tp / (tp + fp);
		rec = (tp + fn) == 0 ? 0 : tp / (tp + fn);
		//fpr = (tn + fp) == 0 ? 0 : fp / (tn + fp);
		//tnr = (tn + fp) == 0 ? 0 : tn / (tn + fp);
		pre_neg = (tn + fn) == 0 ? 0 : tn / (tn + fn)
		rec_neg = (tn + fp) == 0 ? 0 : tn / (tn + fp)
		fscr = 2 * pre * rec / (pre + rec);
		fscr_neg = 2 * pre_neg * rec_neg / (pre_neg + rec_neg);

		if ((tp + fn) != pcnt)  {
			println "numbers don't add up"
		}
			
		println "True Positives\tFalse Positives\tTrue Negatives\tFalse Negatives\tPrecision\tRecall\tFscore\tPrecision_Negative\tRecall_Negative\tFscore_Negative"
		println tp + "\t" + fp + "\t" + tn + "\t" + fn + "\t" + formatter.format(pre) + "\t"  + formatter.format(rec) + "\t" + formatter.format(fscr) + "\t" + formatter.format(pre_neg) + "\t"  + formatter.format(rec_neg) + "\t" + formatter.format(fscr_neg)

	}

	public HybridCCM1(ConfigBundle cb) {
		this.log = LoggerFactory.getLogger(this.class);
		this.config = new PSLConfig(cb);
		this.ds = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, config.dbPath, config.createNewDatastore), this.config.cb);
		this.model = new PSLModel(this, ds);
	}

	public static void main(String[] args) {
		ConfigBundle configBundle = populateConfigBundle(args);
		HybridCCM1 ccm1 = new HybridCCM1(configBundle);
		ccm1.run(configBundle);
    }
}