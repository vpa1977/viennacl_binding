package moa.stream.generators;

import java.util.Random;

import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class ZOrderValidateGenerator extends AbstractOptionHandler implements InstanceStream {

	@Override
	public String getPurposeString() {
		return "Generates 2 perfect clusters to test NN algorithms";
	}

	private static final long serialVersionUID = 1L;

	public static final int NUM_IRRELEVANT_ATTRIBUTES = 17;

	protected static final int originalInstances[][] = { { 0, 0, 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 1, 1, 1 } };

	public IntOption instanceRandomSeedOption = new IntOption("instanceRandomSeed", 'i',
			"Seed for random generation of instances.", 1);

	public IntOption noisePercentageOption = new IntOption("noisePercentage", 'n',
			"Percentage of noise to add to the data.", 10, 0, 100);

	public FlagOption suppressIrrelevantAttributesOption = new FlagOption("suppressIrrelevantAttributes", 's',
			"Reduce the data to only contain 7 relevant binary attributes.");

	protected InstancesHeader streamHeader;

	protected Random instanceRandom;

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// generate header
		this.suppressIrrelevantAttributesOption.set();
		FastVector attributes = new FastVector();
		FastVector binaryLabels = new FastVector();
		binaryLabels.addElement("0");
		binaryLabels.addElement("1");
		int numAtts = 7;
		if (!this.suppressIrrelevantAttributesOption.isSet()) {
			numAtts += NUM_IRRELEVANT_ATTRIBUTES;
		}
		for (int i = 0; i < numAtts; i++) {
			attributes.addElement(new Attribute("att" + (i + 1), binaryLabels));
		}
		FastVector classLabels = new FastVector();
		for (int i = 0; i < 10; i++) {
			classLabels.addElement(Integer.toString(i));
		}
		attributes.addElement(new Attribute("class", classLabels));
		this.streamHeader = new InstancesHeader(
				new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
		this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
		restart();
	}

	@Override
	public long estimatedRemainingInstances() {
		return -1;
	}

	@Override
	public InstancesHeader getHeader() {
		return this.streamHeader;
	}

	@Override
	public boolean hasMoreInstances() {
		return true;
	}

	@Override
	public boolean isRestartable() {
		return true;
	}

	Random rnd = new Random();
	@Override
	public Instance nextInstance() {
		InstancesHeader header = getHeader();
		Instance inst = new DenseInstance(header.numAttributes());
		inst.setDataset(header);
		int selected = this.instanceRandom.nextInt(2);
		for (int i = 0; i < 7; i++) {
				inst.setValue(i, originalInstances[selected][i] + rnd.nextDouble()/10);
		}
		inst.setClassValue(selected);
		return inst;
	}

	@Override
	public void restart() {
		this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
	}
}
