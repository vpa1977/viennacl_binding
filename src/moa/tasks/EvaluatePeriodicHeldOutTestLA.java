/*
 *    EvaluatePeriodicHeldOutTest.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *    @author Ammar Shaker (shaker@mathematik.uni-marburg.de)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.tasks;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.StringUtils;
import moa.core.TimingUtils;
import moa.evaluation.ClassificationPerformanceEvaluator;
import moa.evaluation.LearningCurve;
import moa.evaluation.LearningEvaluation;
import moa.options.ClassOption;
import moa.options.FileOption;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.streams.CachedInstancesStream;
import moa.streams.InstanceStream;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Task for evaluating a classifier on a stream by periodically testing on a heldout set.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class EvaluatePeriodicHeldOutTestLA extends MainTask {

    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a stream by periodically testing on a heldout set.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", InstanceStream.class,
            "generators.RandomTreeGenerator");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            ClassificationPerformanceEvaluator.class,
            "BasicClassificationPerformanceEvaluator");

    public IntOption testSizeOption = new IntOption("testSize", 'n',
            "Number of testing examples.", 1000000, 0, Integer.MAX_VALUE);

    public IntOption trainSizeOption = new IntOption("trainSize", 'i',
            "Number of training examples, <1 = unlimited.", 0, 0,
            Integer.MAX_VALUE);

    public IntOption trainTimeOption = new IntOption("trainTime", 't',
            "Number of training seconds.", 10 * 60 * 60, 0, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption(
            "sampleFrequency",
            'f',
            "Number of training examples between samples of learning performance.",
            100000, 0, Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FlagOption cacheTestOption = new FlagOption("cacheTest", 'c',
            "Cache test instances in memory.");

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
    	int numberOfCheckCalls = 0;
        double modelTime = 0;
        Classifier learner = (Classifier) getPreparedClassOption(this.learnerOption);
        InstanceStream stream = (InstanceStream) getPreparedClassOption(this.streamOption);
        ClassificationPerformanceEvaluator evaluator = (ClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        learner.setModelContext(stream.getHeader());
        long instancesProcessed = 0;
        LearningCurve learningCurve = new LearningCurve("evaluation instances")
        		{

					@Override
					public String entryToString(int entryIndex) {
						// TODO Auto-generated method stub
						return super.entryToString(entryIndex).replace(',', '\t');
					}
        			
        		};
        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        boolean firstDump = true;
        InstanceStream testStream = null;
        int testSize = this.testSizeOption.getValue();
        if (this.cacheTestOption.isSet()) {
            monitor.setCurrentActivity("Caching test examples...", -1.0);
            Instances testInstances = new Instances(stream.getHeader(),
                    this.testSizeOption.getValue());
            while (testInstances.numInstances() < testSize) {
                testInstances.add(stream.nextInstance());
                if (testInstances.numInstances()
                        % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) testInstances.numInstances()
                            / (double) (this.testSizeOption.getValue()));
                }
            }
            testStream = new CachedInstancesStream(testInstances);
        } else {
            //testStream = (InstanceStream) stream.copy();
            testStream = stream;
            /*monitor.setCurrentActivity("Skipping test examples...", -1.0);
            for (int i = 0; i < testSize; i++) {
            stream.nextInstance();
            }*/
        }
        instancesProcessed = 0;
        TimingUtils.enablePreciseTiming();
        double totalTrainTime = 0.0;
        while ((this.trainSizeOption.getValue() < 1
                || instancesProcessed < this.trainSizeOption.getValue())
                && stream.hasMoreInstances() == true) {
            monitor.setCurrentActivityDescription("Training...");
            long instancesTarget = instancesProcessed
                    + this.sampleFrequencyOption.getValue();
            long fixup =0;
            long trainStartTime = System.nanoTime();
            while (instancesProcessed < instancesTarget && stream.hasMoreInstances() == true) {
            	long start = System.nanoTime();
            	Instance i = stream.nextInstance();
            	fixup += System.nanoTime() - start;
                learner.trainOnInstance(i);
                instancesProcessed++;
                if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) (instancesProcessed)
                            / (double) (this.trainSizeOption.getValue()));
                }
            }
            double lastTrainTime = TimingUtils.nanoTimeToSeconds(System.nanoTime()
                    - trainStartTime);
            totalTrainTime += lastTrainTime;
            if (totalTrainTime > this.trainTimeOption.getValue()) {
                break;
            }
	    if (this.cacheTestOption.isSet()) {
                testStream.restart();
            } 
            evaluator.reset();
            long testInstancesProcessed = 0;
            monitor.setCurrentActivityDescription("Testing (after "
                    + StringUtils.doubleToString(
                    ((double) (instancesProcessed)
                    / (double) (this.trainSizeOption.getValue()) * 100.0), 2)
                    + "% training)...");
            long testStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            int instCount = 0 ;
            long count = 0;
            long period = 0;
            
            
            for (instCount = 0; instCount < testSize; instCount++) {
				if (stream.hasMoreInstances() == false) {
					break;
				}
                Instance testInst = (Instance) testStream.nextInstance().copy();
                if (instCount == 0)
                {
                  long modelBuildStart = System.nanoTime();
                	learner.getVotesForInstance(testInst);
                  long modelBuildEnd = System.nanoTime();
                  modelTime = (modelBuildEnd - modelBuildStart)/ 1000000.0; 
                }
                double trueClass = testInst.classValue();
                testInst.setClassMissing();
                
                double[] prediction = learner.getVotesForInstance(testInst);
               
                do // tune to get more or less stable data
                {
	                long start = System.nanoTime();
	                for (int i = 0; i < numberOfCheckCalls ; ++i)
	                	prediction = learner.getVotesForInstance(testInst);
	                long end = System.nanoTime();
	                
	                if ((end - start)/1000000.0 < 30) 
	                {
	                	numberOfCheckCalls += 10;
	                }
	                else
	                {
	                	count +=numberOfCheckCalls;
		                period += (end - start);
		                break;
	                }
                } while (true);
	                
                
                testInst.setClassValue(trueClass);
                evaluator.addResult(testInst, prediction);
                testInstancesProcessed++;
                if (testInstancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) testInstancesProcessed
                            / (double) (testSize));
                }
            }
        	if ( instCount != testSize) {
				break;
			}
        	double final_period = ((double)period / count) / 1000000.0; // msec
            double testTime = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()
                    - testStartTime);
            List<Measurement> measurements = new ArrayList<Measurement>();
            measurements.add(new Measurement("number of attributes",            		
                    testStream.getHeader().numAttributes()-1));
            measurements.add(new Measurement("evaluation instances",            		
                    instancesProcessed));
            measurements.add(new Measurement("test latency", final_period));            
            measurements.add(new Measurement("model build time", modelTime));          
            measurements.add(new Measurement("trainTime", (totalTrainTime/instancesProcessed)));
            Measurement[] performanceMeasurements = evaluator.getPerformanceMeasurements();
            for (Measurement measurement : performanceMeasurements) {
                measurements.add(measurement);
            }
            Measurement[] modelMeasurements = learner.getModelMeasurements();
            for (Measurement measurement : modelMeasurements) {
                measurements.add(measurement);
            }
            learningCurve.insertEntry(new LearningEvaluation(measurements.toArray(new Measurement[measurements.size()])));
            if (immediateResultStream != null) {
                if (firstDump) {
                    immediateResultStream.println(learningCurve.headerToString());
                    firstDump = false;
                }
                immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                immediateResultStream.flush();
            }
            //if (monitor.resultPreviewRequested()) {
             //   monitor.setLatestResultPreview(learningCurve.copy());
            //}
        
        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        return learningCurve;
    }

    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }
}
