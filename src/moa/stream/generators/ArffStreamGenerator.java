package moa.stream.generators;

import java.io.FileReader;
import java.io.IOException;

import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FileOption;
import moa.options.IntOption;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class ArffStreamGenerator  extends AbstractOptionHandler implements InstanceStream {
	
	public FileOption fileOption = new FileOption("file", 'f', "input file", "test.arff", "arff", false);
	public IntOption numberOfInstancesOption= new IntOption("count", 'c', "number of instances", -1);
	private Instances m_header;
	private int m_index;
	private int m_remaining;
	private int m_max_count;
 
	@Override
	public void getDescription(StringBuilder sb, int indent) {
		sb.append("circular arff file");
		
	}

	@Override
	public InstancesHeader getHeader() {
		return new InstancesHeader(m_header);
	}

	@Override
	public long estimatedRemainingInstances() {
		return Long.MAX_VALUE;
	}

	@Override
	public boolean hasMoreInstances() {
		
		return m_remaining < 0 || m_remaining > 0;
	}

	@Override
	public Instance nextInstance() {
		Instance next = m_header.get(m_index);
		if (++m_index == m_header.size())
			m_index = 0;
		if (m_remaining > 0)
			m_remaining --;

		return (Instance)next.copy();
	}

	@Override
	public boolean isRestartable() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public void restart() {
		m_index = 0;
		m_remaining = m_max_count;
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
		ArffReader loader;
		try {
			loader = new ArffReader(new FileReader(fileOption.getValue()));
			m_header = loader.getData();
			m_header.setClass(m_header.attribute("digit"));
			m_remaining = numberOfInstancesOption.getValue();
			m_max_count = m_remaining;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	

}
