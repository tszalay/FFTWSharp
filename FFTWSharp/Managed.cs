// The code in this file is provided courtesy of Tamas Szalay. Some functionality has been added.

// FFTWSharp
// ===========
// Basic C# wrapper for FFTW.
//
// Features
// ============
//    * Unmanaged function calls to main FFTW functions for both single and double precision
//    * Basic managed wrappers for FFTW plans and unmanaged arrays
//    * Test program that demonstrates basic functionality
//
// Notes
// ============
//    * Most of this was written in 2005
//    * Slightly updated since to get it running with Visual Studio Express 2010
//    * If you have a question about FFTW, ask the FFTW people, and not me. I did not write FFTW.
//    * If you have a question about this wrapper, probably still don't ask me, since I wrote it almost a decade ago.

using System;
using System.Runtime.InteropServices;
using System.Numerics;
using System.Threading;

namespace FFTWSharp
{
    #region Single Precision
    /// <summary>
    /// To simplify FFTW memory management
    /// </summary>
    public class fftwf_complexarray
    {
        private IntPtr handle;
        public IntPtr Handle
        { get { return handle; } }

        // The logical length of the array (# of complex numbers, not elements)
        private int length;
        public int Length
        { get { return length; } }

        /// <summary>
        /// Creates a new array of complex numbers
        /// </summary>
        /// <param name="length">Logical length of the array</param>
        public fftwf_complexarray(int length)
        {
            this.length = length;
            this.handle = fftwf.malloc(this.length * 8);
        }

        /// <summary>
        /// Creates an FFTW-compatible array from array of floats, initializes to single precision only
        /// </summary>
        /// <param name="data">Array of floats, alternating real and imaginary</param>
        public fftwf_complexarray(float[] data)
        {
            this.length = data.Length / 2;
            this.handle = fftwf.malloc(this.length * 8);

            this.SetData(data);
        }

        /// <summary>
        /// Creates an FFTW-compatible array from array of Complex numbers
        /// </summary>
        /// <param name="data">Array of Complex numbers</param>
        public fftwf_complexarray(Complex[] data)
        {
            this.length = data.Length;
            this.handle = fftw.malloc(this.length * 16);

            this.SetData(data);
        }

        /// <summary>
        /// Set the data to an array of complex numbers (real + imaginary floats following eachother)
        /// </summary>
        public void SetData(float[] data)
        {
            if (data.Length / 2 != this.length)
                throw new ArgumentException("Array length mismatch!");

            Marshal.Copy(data, 0, handle, this.length * 2);
        }

        /// <summary>
        /// Set the data to an array of complex numbers
        /// </summary>
        public void SetData(Complex[] data)
        {
            if (data.Length != this.length)
                throw new ArgumentException("Array length mismatch!");

            float[] data_in = new float[data.Length * 2];
            for (int i = 0; i < data.Length; i++)
            {
                data_in[2 * i] = (float)data[i].Real;
                data_in[2 * i + 1] = (float)data[i].Imaginary;
            }

            Marshal.Copy(data_in, 0, handle, this.length * 2);
        }

        /// <summary>
        /// Set the data to zeros
        /// </summary>
        public void SetZeroData()
        {
            float[] data_in = new float[this.Length * 2];
            // C# arrays always initialized to 0
            Marshal.Copy(data_in, 0, handle, this.length * 2);
        }

        /// <summary>
        /// Get the data out as Complex numbers
        /// </summary>
        public Complex[] GetData_Complex()
        {
            float[] dataf = new float[length * 2];
            Marshal.Copy(handle, dataf, 0, length * 2);
            Complex[] data = new Complex[length];

            for (int i = 0; i < length; i++)
            {
                data[i] = new Complex(dataf[2 * i], dataf[2 * i + 1]);
            }

            return data;
        }

        /// <summary>
        /// Get the real elements out (imaginary parts are discarded)
        /// </summary>
        public float[] GetData_Real()
        {
            float[] dataf = new float[length * 2];
            Marshal.Copy(handle, dataf, 0, length * 2);
            float[] data = new float[length];

            for (int i = 0; i < length; i++)
            {
                data[i] = dataf[2 * i];
            }

            return data;
        }

        /// <summary>
        /// Get the full array of floats out (alternating real and imaginary)
        /// </summary>
        public float[] GetData_Float()
        {
            float[] dataf = new float[length * 2];
            Marshal.Copy(handle, dataf, 0, length * 2);

            return dataf;
        }

        ~fftwf_complexarray()
        {
            fftwf.free(handle);
        }
    }

    /// <summary>
    /// Creates, stores, and destroys fftw plans
    /// </summary>
    public class fftwf_plan
    {
        static Mutex FFTW_Lock = new Mutex();

        protected IntPtr handle;
        public IntPtr Handle
        { get { return handle; } }

        public void Execute()
        {
            fftwf.execute(handle);
        }

        ~fftwf_plan()
        {
            fftwf.destroy_plan(handle);
        }

        #region Plan Creation
        //Complex<->Complex transforms
        public static fftwf_plan dft_1d(int n, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_1d(n, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_2d(int nx, int ny, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_2d(nx, ny, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_3d(int nx, int ny, int nz, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_3d(nx, ny, nz, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft(int rank, int[] n, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft(rank, n, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        //Real->Complex transforms
        public static fftwf_plan dft_r2c_1d(int n, fftwf_complexarray input, fftwf_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_r2c_1d(n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_r2c_2d(int nx, int ny, fftwf_complexarray input, fftwf_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_r2c_2d(nx, ny, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_r2c_3d(int nx, int ny, int nz, fftwf_complexarray input, fftwf_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_r2c_3d(nx, ny, nz, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_r2c(int rank, int[] n, fftwf_complexarray input, fftwf_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_r2c(rank, n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        //Complex->Real
        public static fftwf_plan dft_c2r_1d(int n, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_c2r_1d(n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_c2r_2d(int nx, int ny, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_c2r_2d(nx, ny, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_c2r_3d(int nx, int ny, int nz, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_c2r_3d(nx, ny, nz, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan dft_c2r(int rank, int[] n, fftwf_complexarray input, fftwf_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.dft_c2r(rank, n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        //Real<->Real
        public static fftwf_plan r2r_1d(int n, fftwf_complexarray input, fftwf_complexarray output, fftw_kind kind, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.r2r_1d(n, input.Handle, output.Handle, kind, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan r2r_2d(int nx, int ny, fftwf_complexarray input, fftwf_complexarray output, fftw_kind kindx, fftw_kind kindy, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.r2r_2d(nx, ny, input.Handle, output.Handle, kindx, kindy, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan r2r_3d(int nx, int ny, int nz, fftwf_complexarray input, fftwf_complexarray output,
            fftw_kind kindx, fftw_kind kindy, fftw_kind kindz, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.r2r_3d(nx, ny, nz, input.Handle, output.Handle,
                kindx, kindy, kindz, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftwf_plan r2r(int rank, int[] n, fftwf_complexarray input, fftwf_complexarray output,
            fftw_kind[] kind, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftwf_plan p = new fftwf_plan();
            p.handle = fftwf.r2r(rank, n, input.Handle, output.Handle,
                kind, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }
        #endregion
    }
    #endregion

    #region Double Precision
    /// <summary>
    /// So FFTW can manage its own memory nicely
    /// </summary>
    public class fftw_complexarray
    {
        private IntPtr handle;
        public IntPtr Handle
        { get { return handle; } }

        private int length;
        public int Length
        { get { return length; } }

        /// <summary>
        /// Creates a new array of complex numbers
        /// </summary>
        /// <param name="length">Logical length of the array</param>
        public fftw_complexarray(int length)
        {
            this.length = length;
            this.handle = fftw.malloc(this.length * 16);
        }

        /// <summary>
        /// Creates an FFTW-compatible array from array of doubles
        /// </summary>
        /// <param name="data">Array of doubles, alternating real and imaginary</param>
        public fftw_complexarray(double[] data)
        {
            this.length = data.Length / 2;
            this.handle = fftw.malloc(this.length * 16);

            this.SetData(data);
        }

        /// <summary>
        /// Creates an FFTW-compatible array from array of Complex numbers
        /// </summary>
        /// <param name="data">Array of Complex numbers</param>
        public fftw_complexarray(Complex[] data)
        {
            this.length = data.Length;
            this.handle = fftw.malloc(this.length * 16);

            this.SetData(data);
        }

        /// <summary>
        /// Set the data to an array of complex numbers
        /// </summary>
        public void SetData(double[] data)
        {
            if (data.Length / 2 != this.length)
                throw new ArgumentException("Array length mismatch!");

            Marshal.Copy(data, 0, handle, this.length * 2);
        }

        /// <summary>
        /// Set the data to an array of complex numbers
        /// </summary>
        public void SetData(Complex[] data)
        {
            if (data.Length != this.length)
                throw new ArgumentException("Array length mismatch!");

            double[] data_in = new double[data.Length * 2];
            for (int i = 0; i < data.Length; i++)
            {
                data_in[2 * i] = data[i].Real;
                data_in[2 * i + 1] = data[i].Imaginary;
            }

            Marshal.Copy(data_in, 0, handle, this.length * 2);
        }

        /// <summary>
        /// Set the data to zeros.
        /// </summary>
        public void SetZeroData()
        {
            double[] data_in = new double[this.Length * 2];
            // C# arrays always initialized to 0
            Marshal.Copy(data_in, 0, handle, this.length * 2);
        }

        /// <summary>
        /// Get the data out
        /// </summary>
        /// <returns></returns>
        public Complex[] GetData_Complex()
        {
            double[] datad = new double[length * 2];
            Marshal.Copy(handle, datad, 0, length * 2);
            Complex[] data = new Complex[length];

            for (int i = 0; i < length; i++)
            {
                data[i] = new Complex(datad[2 * i], datad[2 * i + 1]);
            }

            return data;
        }

        public double[] GetData_Real()
        {
            double[] datad = new double[length * 2];
            Marshal.Copy(handle, datad, 0, length * 2);
            double[] data = new double[length];

            for (int i = 0; i < length; i++)
            {
                data[i] = datad[2 * i];
            }

            return data;
        }

        /// <summary>
        /// Get the data out
        /// </summary>
        /// <returns></returns>
        public double[] GetData_Double()
        {
            double[] datad = new double[length * 2];
            Marshal.Copy(handle, datad, 0, length * 2);

            return datad;
        }

        ~fftw_complexarray()
        {
            fftw.free(handle);
        }
    }

    /// <summary>
    /// Creates, stores, and destroys fftw plans
    /// </summary>
    public class fftw_plan
    {
        static Mutex FFTW_Lock = new Mutex();

        protected IntPtr handle;
        public IntPtr Handle
        { get { return handle; } }

        public void Execute()
        {
            fftw.execute(handle);
        }

        ~fftw_plan()
        {
            fftw.destroy_plan(handle);
        }

        #region Plan Creation
        //Complex<->Complex transforms
        public static fftw_plan dft_1d(int n, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_1d(n, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan dft_2d(int nx, int ny, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_2d(nx, ny, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();
            
            return p;
        }

        public static fftw_plan dft_3d(int nx, int ny, int nz, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_3d(nx, ny, nz, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();
            
            return p;
        }

        public static fftw_plan dft(int rank, int[] n, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft(rank, n, input.Handle, output.Handle, direction, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        //Real->Complex transforms
        public static fftw_plan dft_r2c_1d(int n, fftw_complexarray input, fftw_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_r2c_1d(n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan dft_r2c_2d(int nx, int ny, fftw_complexarray input, fftw_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_r2c_2d(nx, ny, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan dft_r2c_3d(int nx, int ny, int nz, fftw_complexarray input, fftw_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_r2c_3d(nx, ny, nz, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan dft_r2c(int rank, int[] n, fftw_complexarray input, fftw_complexarray output, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_r2c(rank, n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        //Complex->Real
        public static fftw_plan dft_c2r_1d(int n, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_c2r_1d(n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan dft_c2r_2d(int nx, int ny, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_c2r_2d(nx, ny, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan dft_c2r_3d(int nx, int ny, int nz, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_c2r_3d(nx, ny, nz, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan dft_c2r(int rank, int[] n, fftw_complexarray input, fftw_complexarray output, fftw_direction direction, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.dft_c2r(rank, n, input.Handle, output.Handle, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        //Real<->Real
        public static fftw_plan r2r_1d(int n, fftw_complexarray input, fftw_complexarray output, fftw_kind kind, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.r2r_1d(n, input.Handle, output.Handle, kind, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan r2r_2d(int nx, int ny, fftw_complexarray input, fftw_complexarray output, fftw_kind kindx, fftw_kind kindy, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.r2r_2d(nx, ny, input.Handle, output.Handle, kindx, kindy, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan r2r_3d(int nx, int ny, int nz, fftw_complexarray input, fftw_complexarray output,
            fftw_kind kindx, fftw_kind kindy, fftw_kind kindz, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.r2r_3d(nx, ny, nz, input.Handle, output.Handle,
                kindx, kindy, kindz, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }

        public static fftw_plan r2r(int rank, int[] n, fftw_complexarray input, fftw_complexarray output,
            fftw_kind[] kind, fftw_flags flags)
        {
            FFTW_Lock.WaitOne();
            fftw_plan p = new fftw_plan();
            p.handle = fftw.r2r(rank, n, input.Handle, output.Handle,
                kind, flags);
            FFTW_Lock.ReleaseMutex();

            return p;
        }
        #endregion
    }
    #endregion
}