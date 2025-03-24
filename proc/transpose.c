//This code is released under the GPL license

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define BLOCKSIZE 10240 //Read file in chunks of 10kb
#define MAXCOLS 1000 //Allow for this many columns by default
#define MAXROWS 1000 //Allow for this many rows by default
#define FIELDMAX 64 //Allow for this many characters in a field

typedef struct {
	unsigned int fill;
	unsigned int size;
	size_t unit;
	void* data; 
} buffer;

//Function to Dispay Version Information//////////////
void Version() { fprintf(stderr,"\ntranspose: Version 2.0 - December 2010 - Dr. Alex Sheppard (www.das-computer.co.uk)\n\n"); }

//// Function Giving Help and Information ////////////////////////////
void Usage() 
{ 

        fprintf(stderr,"\n\n \
                        \r\tDescription:\n \
			\r\tThis software is released under the GPL license\n \
			\r\tReshapes delimited text data - amongst other things, it can transpose a matrix of plain text data.\n \
			\r\tThe input file (optional, otherwise stdin is used) should be the last argument provided\n\n \
			\r\tOptions:\n \
                        \r\t-v or --version\t\tdisplays version information\n \
                        \r\t-h or --help\t\tdisplays this helpful information\n \
                        \r\t-b or --block <integer>\t\tspecify the blocksize (in bytes) for reading the input (default is 10kb)\n \
                        \r\t--fsep <character>\t\tsets field separator to specified character. Defaults is \\t\n \
                        \r\t--lsep <character>\t\tsets line separator to specified character. Default is \\n\n \
			\r\t--win \t\tsets the line separator to the 'Windows Newline' \\r\\n \
			\r\t--fieldmax or -f <integer>\t\tsets the number of characters to allow for in each field (default is 64)\n \
			\r\t--input or -i <integer>x<integer>[+<integer>+<integer>]\t\tspecificies size and cropping of the input. Dimension order is rows, columns\n \
			\r\t--output or -o <integer>x<integer>\t\tspecificies dimensions for the output. Default is determined from the input dimensions and whether we are transposing.\n \
			\r\t--limit or -l <integer>x<integer>\t\twhen not input dimensions, allow for this size (rowsxcolumns) matrix. Default is 1000x1000\n \
			\r\t--transpose or -t\t\tindicates that the output matrix is to be filled in columnwise. When output dimensions are not specified (or equal the transpose of the input dimensions) the output is an exact transpose of the input.\n \
			\r");

}

buffer* Buffer_New(unsigned int size, size_t unit)
{

	buffer* B;
	B = (buffer*)malloc(sizeof(buffer));
	B->fill = 0;
	B->size = size;
	B->unit = unit;
	B->data = malloc(size*unit);

	return B;

}

int Buffer_Append(buffer* B, void* src, unsigned int length)
{

	unsigned int copied;
	unsigned int a;
	
	copied = ((a = B->size - B->fill) >= length) ? length : a;
	memcpy(B->data + B->fill*B->unit,src,copied*B->unit);
	B->fill += copied;

	//Return the amount of space remaining ( < 0 indicates not all items were copied)
	return (B->size - B->fill - (length - copied)); 

}

//Use when you have a buffer where buffer.data is an array of buffers.
//This function appends a buffer to the array, and returns the free space remaining in the parent buffer.
//The data is COPIED - a new block is allocated and the data from src->data is copied to it. The src buffer can be re-used.
int Buffer_AppendBuffer(buffer* B, buffer* src)
{

	if (B->fill >= B->size)
	{

		return -1;
	
	} else {

		((buffer*)B->data)[B->fill].size = src->fill;
		((buffer*)B->data)[B->fill].fill = src->fill;
		((buffer*)B->data)[B->fill].unit = src->unit;

		((buffer*)B->data)[B->fill].data = malloc(src->unit*src->fill);
		memcpy(((buffer*)B->data)[B->fill].data,src->data,src->fill*src->unit);

		B->fill++;

		return (B->size - B->fill);

	}

}

int Buffer_AppendDataPtr (buffer* B, buffer* src, char terminate)
{

	void* p = 0;

	if (B->fill >= B->size)
	{

		return -1;
	
	} else if (terminate) {

		p = calloc(src->fill+1,src->unit);

	} else {

		p = malloc(src->fill*src->unit);

	}

	memcpy(p,src->data,src->fill*src->unit);
	memcpy(B->data+B->fill*B->unit,&p,sizeof(void*));
	B->fill++;

	return (B->size - B->fill);

}

unsigned int RefillInput(buffer* I, unsigned int n, FILE* fdr)
{

	if (I->fill >= I->size)
	{

		//The buffer was filled last time
		memmove(I->data,I->data+I->size - n,n);  
		I->fill = n + fread(I->data+n,sizeof(char),I->size-n,fdr);

	} else if (!I->fill) {

		//This is done on first call to Refill
		I->fill = fread(I->data,sizeof(char),I->size,fdr);
		
	} else {

		I->fill = 0;

	}

	return (I->fill < I->size) ? I->fill : I->fill - n;

}

int TestForSep(char* str, char fsep[], size_t fn, char lsep[], size_t ln)
{

	if (!strncmp(str,fsep,fn))
	{

		return -1;

	} else if (!strncmp(str,lsep,ln)) {

		return -2;

	} else {

		return 0;

	}

}

buffer* ReadMatrix (FILE* fdr, unsigned int blocksize, unsigned short fieldmax, char fsep[3], char lsep[3], unsigned int* dims, unsigned int limit[2])
{

	buffer* input_buffer; //A buffer where data is a char*
	buffer* field_buffer; //A buffer where data is a char*
	buffer* row_buffer; //A buffer where data is a char**
	buffer* Matrix; //A buffer where data is a buffer*
	unsigned int maxdims[2];
	unsigned int skip[2];
	size_t sepsize[3];
	char lineskip = 0, fieldskip = 0, fileskip = 0, byteskip = 0;
	unsigned int i, a, cols = 0;
	char* byteptr;

	maxdims[0] = dims[0] ? dims[0] : limit[0];
	maxdims[1] = dims[1] ? dims[1] : limit[1];
	skip[0] = dims[2];
	skip[1] = dims[3];

	sepsize[0] = strlen(fsep);
	sepsize[1] = strlen(lsep);
	sepsize[2] = sepsize[0] > sepsize[1] ? sepsize[0] : sepsize[1];

	input_buffer = Buffer_New(blocksize,sizeof(char));
	field_buffer = Buffer_New(fieldmax,sizeof(char));
	row_buffer = Buffer_New(maxdims[1],sizeof(char*));
	Matrix = Buffer_New(maxdims[0],sizeof(buffer)); //A buffer containing pointers to rows (each row is a buffer too).

	while ((a = RefillInput(input_buffer,sepsize[2]-1,fdr)) && Matrix->fill < maxdims[0])
	{

		for (i=0; i<a && Matrix->fill < maxdims[0]; i++)
		{

			if (byteskip)
			{ 

				byteskip--;
				continue;

			}

			//Test end of field for a separator
			switch (TestForSep((byteptr = (char*)(input_buffer->data+i)),fsep,sepsize[0],lsep,sepsize[1]))
			{

				case -1: //We matched fsep
					byteskip = strlen(fsep) - 1;
					if (skip[0])
					{

						//Skipping whole row - nothing to do here
						continue;

					} else if (skip[1]) {

						//Skipping this column. Decrement skip[1] - we might want the later columns 
						skip[1]--;

					} else if (!lineskip && !fieldskip) { 

						//Row not already finished, save the current field 
						if (!Buffer_AppendDataPtr(row_buffer,field_buffer,1))
						{

							if (!Buffer_AppendBuffer(Matrix,row_buffer)) { fileskip = 1; } 
							cols = row_buffer->fill > cols ? row_buffer->fill : cols;
							row_buffer->fill = 0;
							lineskip = 1;

						}
						field_buffer->fill = 0;

					}
					fieldskip = 0;
					break;
	
				case -2: //we matched lsep
					byteskip = strlen(lsep) - 1;
					if (skip[0])
					{

						skip[0]--;

					} else if (!lineskip && !fieldskip) { 

						//Row not already finished - save it
						Buffer_AppendDataPtr(row_buffer,field_buffer,1);
						if (!Buffer_AppendBuffer(Matrix,row_buffer)) { fileskip = 1; } 
						cols = row_buffer->fill > cols ? row_buffer->fill : cols;
						row_buffer->fill = 0;
						field_buffer->fill = 0;

					}
					skip[1] = dims[3]; //resest column skip ready for next row.
					fieldskip = 0;
					lineskip = 0;
					break;

				default:
					if (skip[1] || skip[0] || fieldskip || lineskip)
					{

						continue;

					} else if (!Buffer_Append(field_buffer,byteptr,1)) {

						if (!Buffer_AppendDataPtr(row_buffer,field_buffer,1))
						{

							if (!Buffer_AppendBuffer(Matrix,row_buffer)) { fileskip = 1; } 
							cols = row_buffer->fill > cols ? row_buffer->fill : cols;
							row_buffer->fill = 0;
							lineskip = 1;

						} 
						fieldskip = 1;
						field_buffer->fill = 0;

					}
					break;

			}

		}

		if (fileskip) { break; }

	}

	//Check for unterminated field - pretend we met a field separator
	if (!skip[1] && !skip[0] && !fileskip && !fieldskip && field_buffer->fill)
	{

		if (!Buffer_AppendDataPtr(row_buffer,field_buffer,1))
		{

			Buffer_AppendBuffer(Matrix,row_buffer);
			row_buffer->fill = 0;
			lineskip = 1;

		}
		field_buffer->fill = 0;

	}

	//Check for unterminated line - attempt to add row to matrix
	if (!skip[1] && !skip[0] && !fileskip && row_buffer->fill)
	{ 

		Buffer_AppendBuffer(Matrix,row_buffer);
		row_buffer->fill = 0;
		field_buffer->fill = 0;

	}

	//Free some memory
	free(row_buffer->data);
	free(row_buffer);
	free(input_buffer->data);
	free(input_buffer);
	free(field_buffer->data);
	free(field_buffer);

	//Populate dims?
	dims[0] = dims[0] ? dims[0] : Matrix->fill;
	dims[1] = dims[1] ? dims[1] : cols;
	
	return Matrix;

}

char* GetField(buffer* Matrix, unsigned int row, unsigned int column)
{

	if (row >= Matrix->fill)
	{

		return 0;

	} else if (column >= ((buffer*)Matrix->data)[row].fill) {

		return 0;

	} else {

		return ((char**)((buffer*)Matrix->data)[row].data)[column];
	}

}


//FIXME - how much speed might we gain by writing in blocks rather than field by field?  
int PrintMatrix(FILE* fdw, buffer* Matrix, char fsep[3], char lsep[3], unsigned int input[],  unsigned int output[], char transpose)
{

	unsigned int ri, ci, a;
	unsigned long fi;
	char* fieldstr;
	char samedims;
	char empty[1] = "\0";

	samedims = (output[0] == input[0] && output[1] == input[1]) || (transpose && (output[0] == input[1] && output[1] == input[0]));

	for (ri=0; ri<output[0]; ri++)
	{

		for (ci=0; ci<output[1]; ci++)
		{

			if (samedims && transpose)
			{

				fieldstr = GetField(Matrix,ci,ri); 

			} else if (samedims) {

				fieldstr = GetField(Matrix,ri,ci); 

			} else if (transpose) { 
			
				fi = ri*output[1] + ci;
				a = fi/input[0];
				fieldstr = GetField(Matrix,(fi-a*input[0]),a);

			} else {

				fi = ri*output[1] + ci;
				a = fi/input[1];
				fieldstr = GetField(Matrix,a,(fi-a*input[1]));

			}

			fieldstr = fieldstr ? fieldstr : empty;

			if (ci >= output[1]-1) 
			{

				fprintf(fdw,"%s%s",fieldstr,lsep);

			} else {

				fprintf(fdw,"%s%s",fieldstr,fsep);

			}

		}

	}

	return 0;

}

void FreeMatrix (buffer* Matrix)
{

	unsigned int ri, ci;

	for (ri=0; ri<Matrix->fill; ri++)
	{

		for (ci=0; ci<((buffer*)Matrix->data)[ri].fill; ci++)
		{

			free(((char**)((buffer*)Matrix->data)[ri].data)[ci]);

		}

		free(((buffer*)Matrix->data)[ri].data);

	}

	free(Matrix->data);
	free(Matrix);

}


int main (int argc, char **argv)
{
	
	buffer* Matrix;
	FILE* fdr = stdin; //read file
	FILE* fdw = stdout; //write file
	unsigned int blocksize = BLOCKSIZE; //read input in chunks of this size
	char fsep[3] = {9,0,0}; //field separator
	char lsep[3] = {10,0,0}; //line separator
	unsigned short fieldmax = FIELDMAX; //maximum characters in field
	unsigned int input[4] = {0,0,0,0}; //dimensions of input matrix e.g. 4x4+1+1 would mean: use Matrix[2-5][2-5]
	unsigned int output[2] = {0,0}; //dimensions requested for output.
	unsigned int limit[2] = {0,0}; //limit dimensions (when exact input dimensions are not known
	char transpose = 0; //flag to transpose or not.
	unsigned int a; //throwaway variable for loops etc.

	////Deal With Options////////////////////////////////////////////
	for (a=1 ; a<argc ; a++) //loop through arguments
	{

		if ( (!strcmp(argv[a],"-h")) || (!strcmp(argv[a],"--help")) ) { Usage(); exit(0); }
		if ( (!strcmp(argv[a],"-v")) || (!strcmp(argv[a],"--version")) ) { Version(); exit(0); } 
		if (!strcmp(argv[a],"--block") || !strcmp(argv[a],"-b")) { blocksize = atoi(argv[a+1]); a++; continue; }
		if (!strcmp(argv[a],"--fieldmax") || !strcmp(argv[a],"-f")) { fieldmax = atoi(argv[a+1]); a++; continue; }
		if (!strcmp(argv[a],"--fsep")) 
		{ 
		
			strncpy(fsep,argv[a+1],2);
			a++; 
			continue;
	
		}
		if (!strcmp(argv[a],"--lsep")) 
		{ 
		
			strncpy(lsep,argv[a+1],2);
			a++; 
			continue;
	
		}
		if (!strcmp(argv[a],"--win")) { strcpy(lsep,"\r\n"); continue; }
		if (!strcmp(argv[a],"--transpose") || !strcmp(argv[a],"-t")) { transpose = 1; continue; }
		if (!strcmp(argv[a],"--input") || !strcmp(argv[a],"-i")) 
		{ 

			if (sscanf(argv[a+1],"%ix%i+%i+%i",input,input+1,input+2,input+3) == 4) 
			{
	
				a++;
				continue;

			} else if (sscanf(argv[a+1],"%ix%i",input,input+1) == 2) {

				a++;
				continue;

			} else {

				fprintf(stderr,"Incorrect input dimensions.\n");
				exit(1);

			}

		}
		if (!strcmp(argv[a],"--output") || !strcmp(argv[a],"-o")) 
		{ 

			if (sscanf(argv[a+1],"%ix%i",output,output+1) != 2) 
			{

				fprintf(stderr,"Incorrect output dimensions.\n");
				exit(1);

			} else {

				a++;
				continue;

			} 

		}
		if (!strcmp(argv[a],"--limit") || !strcmp(argv[a],"-l")) 
		{ 

			if (sscanf(argv[a+1],"%ix%i",limit,limit+1) != 2) 
			{

				fprintf(stderr,"Incorrect limit dimensions.\n");
				exit(1);

			} else {

				a++;
				continue;

			} 

		}

		//Last argument is assumed to be input filename - an unrecognised argument that is not the last argument is an error.
		if (a != argc-1)
		{

			fprintf(stderr,"Unrecognised option / argument: %s\n",argv[a]);
			exit(1);

		} else if ( !(fdr=fopen(argv[a],"rb")) ) {

			fprintf(stderr,"Unable to Open File %s\n\n",argv[a]);
			exit(1);

		} 

	}

	//Process dimension limits
	limit[0] = limit[0] ? limit[0] : MAXROWS;
	limit[1] = limit[1] ? limit[1] : MAXCOLS;

	//Read data into Matrix
	Matrix = ReadMatrix(fdr,blocksize,fieldmax,fsep,lsep,&input[0],limit);
	fclose(fdr);

	//Print Matrix
	if (!output[0]) { output[0] = transpose ? input[1] : input[0]; }
	if (!output[1]) { output[1] = transpose ? input[0] : input[1]; }

	PrintMatrix(fdw,Matrix,fsep,lsep,input,output,transpose);
	fclose(fdw);

	FreeMatrix(Matrix);
	
	return 0;

}

