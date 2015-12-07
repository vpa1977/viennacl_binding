package org.moa.opencl.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public abstract class AbstractUtil {
	protected StringBuffer loadKernel(String name) {
		String line;
		InputStream is = getClass().getResourceAsStream(name);
		StringBuffer data = new StringBuffer();
		
		BufferedReader r = new BufferedReader(new InputStreamReader(is));
		try {
			while ( (line = r.readLine()) != null)
				data.append(line).append('\n');
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return data;
	}

}
