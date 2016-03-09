import java.io.File;
import java.io.FileOutputStream;

import moa.streams.generators.RandomRBFGenerator;
import weka.core.converters.ArffSaver;

public class CreateTestArff {
	public static void main(String[] args) throws Throwable
	{
		RandomRBFGenerator generator = new RandomRBFGenerator();
		generator.prepareForUse();
		ArffSaver saver = new ArffSaver();
		saver.setStructure(generator.getHeader());
		saver.setRetrieval(ArffSaver.INCREMENTAL);
		saver.setDestination(new FileOutputStream(new File("L:/random_rbf.arff")));
		for (int i = 0;i <483328; ++i )
		{
			saver.writeIncremental(generator.nextInstance());
		}
	}
}
