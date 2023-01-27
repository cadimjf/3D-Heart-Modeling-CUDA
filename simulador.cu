#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// discretization in m
#define dx 0.001
//x, y and z dimensiona
#define X_SIZE 100
#define Y_SIZE 100
#define Z_SIZE 100
#define DIM 3
// volume em metros cúbicos
#define volume dx*(X_SIZE-1)*dx*(Y_SIZE-1)*dx*(Z_SIZE-1)
//densidade de massa (kg/m^3):
#define ro 100
// a massa é encontrada multiplicando o volume pela densidade
#define mass volume*ro
// força máxima por área, em pascal (N/m^2)
#define pressure 13500
#define HEALTHY 0
#define ISCHEMIC 1
#define DEAD	3
//força máxima que um elemento faz ao ser ativada pelo AP -> multiplica a pressão pela área formada pelo quadrado (dx X dx) e divide por quatro, representando os 4 elmentos q formam um quadrado
#define tForce pressure*(dx*dx)/4
#define get1DIndex(i, j, k) i*X_SIZE*Y_SIZE + j*Y_SIZE + k
// spring coefficient N/m
#define ks 15.0
// Overdamped (ζ > 1): The system returns (exponentially decays) to equilibrium without oscillating. Larger values of the damping ratio ζ return to equilibrium more slowly.
// Critically damped (ζ = 1): The system returns to equilibrium as quickly as possible without oscillating. This is often desired for the damping of systems such as doors.
// Underdamped (0 < ζ < 1): The system oscillates (at reduced frequency compared to the undamped case) with the amplitude gradually decreasing to zero.
// Undamped (ζ = 0): The system oscillates at its natural resonant frequency (ωo).
//l = kd/(2*sqrt(m*ks));
// damping coefficient Kg/s
// float kd = 10*2*sqrt(mass*ks);
#define kd 2*sqrt(mass*ks)
// float kd = 0.5* 2*sqrt(mass*ks);
// float kd = 0.0* 2*sqrt(mass*ks);
// float kd=0.0001;
// preserving volume coefficient
#define  kv 0.05

//s
#define initialPace 0.001
//s
#define stimulusPeriod 4.0

#define volIni dx*dx*dx
typedef struct str_elem{
    float pos[DIM];
    float vel[DIM];
    int stateV;
    int stateF;
    float cellTime;
    int paceMaker;
    int condition;
}typ_elem;
int contFreq=0;
float areaIni=0.0;

/**
 */
void iniElement(typ_elem *elem_new, typ_elem *elem_old, int i, int j, int k){
	for(int l=0;l<DIM;l++){
		elem_new->vel[l] = 0.0;
		elem_old->vel[l] = 0.0;
	}
	elem_new->pos[0] = i*dx;
	elem_new->pos[1] = j*dx;
	elem_new->pos[2] = k*dx;
	
	elem_old->pos[0] = i*dx;
	elem_old->pos[1] = j*dx;
	elem_old->pos[2] = k*dx;
	
	elem_old->stateV	= 0;
	elem_old->stateF	= 0;
	elem_old->cellTime	= 0.0;
	elem_old->paceMaker	= 0;
	elem_old->condition	= HEALTHY;
	
	elem_new->stateV	= 0;
	elem_new->stateF	= 0;
	elem_new->cellTime	= 0.0;
	elem_new->paceMaker	= 0;
	elem_new->condition	= HEALTHY;
}
/**
 * 
 */
__device__ float norm (float v[DIM]){
	return sqrt(pow(v[0],2) + pow(v[1],2) + pow(v[2],2));
}
/**
 */
__device__ float dotProduct(float a[DIM], float b[DIM]){
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
__device__ float det3(float a[DIM][DIM]){
	return a[0][0]*((a[1][1]*a[2][2]) - (a[2][1]*a[1][2])) -a[0][1]*(a[1][0]*a[2][2] - a[2][0]*a[1][2]) + a[0][2]*(a[1][0]*a[2][1] - a[2][0]*a[1][1]);
}

/**
 */
__device__ float getVolCube(float a[DIM], float b[DIM], float c[DIM], float d[DIM]){
	float m[DIM][DIM];
	for( int l=0; l<DIM; l++){
		m[0][l] = b[l] - a[l];
		m[1][l] = c[l] - a[l];
		m[2][l] = d[l] - a[l];
	}
	return fabs(det3(m));
}
/**
 */
__device__ float getDistance(float a[DIM], float b[DIM]){
	float aux[DIM];
	aux[0] = b[0] - a[0];
	aux[1] = b[1] - a[1];
	aux[2] = b[2] - a[2];
	return norm(aux);
}
/**
 */
__device__ void getDampingForce(int i,int j, int k, typ_elem* mesh, float force[DIM]){
	
	for(int l=0; l<DIM; l++)
		force[l] += -(mesh[get1DIndex(i,j,k)].vel[l]*kd);
}
/**
 */
__device__ float getVolRestriction(int i, int i1, int j, int j1, int k, int k1, typ_elem* mesh){
	float vol = getVolCube(		mesh[get1DIndex(i,j,k)].pos, 
					mesh[get1DIndex(i1,j,k)].pos, 
					mesh[get1DIndex(i,j1,k)].pos,
					mesh[get1DIndex(i,j,k1)].pos);
	return (vol - volIni)/volIni;
}
/**
 */
__device__ void getBaricenter(int i, int i1, int j, int j1, int k, int k1, typ_elem* mesh, float b[DIM]){
	for(int l=0;l<DIM;l++){
		b[l] = (
			mesh[get1DIndex(i,j,k)].pos[l]		+ mesh[get1DIndex(i1,j,k)].pos[l]	+ mesh[get1DIndex(i1,j1,k)].pos[l]	+ mesh[get1DIndex(i,j1,k)].pos[l] +
			mesh[get1DIndex(i,j1,k1)].pos[l]	+ mesh[get1DIndex(i,j,k1)].pos[l]	+ mesh[get1DIndex(i1,j,k1)].pos[l]	+ mesh[get1DIndex(i1,j1,k1)].pos[l]
			)/8.0;
	}
}
/**
 */
__device__ float getNeighborCube_VolPreservingForce(int i, int i1, int j, int j1, int k, int k1, typ_elem* mesh, float force[DIM]){
	float r = getVolRestriction(i, i1, j, j1, k, k1, mesh);
	float b[DIM];	float dir;
	getBaricenter(i, i1, j, j1, k, k1, mesh, b);
	int i1d = get1DIndex(i,j,k);
	float n =getDistance( b, mesh[i1d].pos);
	for(int l=0;l<DIM;l++){
		dir		= (mesh[i1d].pos[l] -b[l])/n;
		force[l]	+= -kv*r*dir;
	}
}

/**
 */
__device__ void getHookeForce(int i, int j, int k, typ_elem* mesh, float force[DIM], int ii, int jj, int kk){
	float delta[DIM];
	int i1d		= get1DIndex(i,j,k);
	int i1d2	= get1DIndex(ii,jj,kk);
	delta[0] = (ii-i)*dx - (mesh[i1d2].pos[0] - mesh[i1d].pos[0]); 
	delta[1] = (jj-j)*dx - (mesh[i1d2].pos[1] - mesh[i1d].pos[1]); 
	delta[2] = (kk-k)*dx - (mesh[i1d2].pos[2] - mesh[i1d].pos[2]);
	for(int l=0; l<DIM; l++){
		force[l] += -delta[l]*ks;
	}
}

/**
 */
__device__ float getMeshVol(typ_elem* mesh){
	float vol = 0.0, hm=0.0;
	int cont=0;
	float a1=0.0, a2=0.0;
	for(int i=0; i<X_SIZE-1; i++){
		for(int j=0; j<Y_SIZE-1; j++){
			//Obtem a area da fave z=0
			a1 += getDistance(mesh[get1DIndex(i,j,0)].pos, mesh[get1DIndex(i+1,j,0)].pos)*getDistance(mesh[get1DIndex(i,j,0)].pos, mesh[get1DIndex(i,j+1,0)].pos);
			//Obtem a area da fave z=Z_SIZE-1
			a2 += getDistance(mesh[get1DIndex(i,j,Z_SIZE-1)].pos, mesh[get1DIndex(i+1,j,Z_SIZE-1)].pos)*getDistance(mesh[get1DIndex(i,j,Z_SIZE-1)].pos, mesh[get1DIndex(i,j+1,Z_SIZE-1)].pos);
			// 	encontra a altura media
			hm += getDistance(mesh[get1DIndex(i,j,0)].pos, mesh[get1DIndex(i,j,Z_SIZE-1)].pos);
			cont++;
		}
	}

	hm = hm/cont;
	vol = ((a1+a2)/2) *hm;
	return vol;
	
}
/**
 * return the force in Newton 
*/
__device__ float getF(typ_elem* elem){
	float f=0.0;
	if(elem->condition==HEALTHY){
		switch(elem->stateF){
			case 0:
				f=0.0;
			break;
			case 1:
				f = (elem->cellTime-0.050f)/0.080f;
			break;
			case 2:
				f = 1.0;
			break;
			case 3:
				f= 1.0- (elem->cellTime-0.230f)/0.080f;
			break;
			default:
				f= 0.0;
		}
	}
	if(elem->condition==ISCHEMIC){
		switch(elem->stateF){
			case 0:
				f=0.0;
			break;
			case 1:
				f = (elem->cellTime-0.050f)/0.040f;
			break;
			case 2:
				f = 1.0;
			break;
			case 3:
				f= 1.0- (elem->cellTime-0.140f)/0.035f;
			break;
			default:
				f= 0.0;
		}
	}
	if(elem->condition==DEAD) f=0.0;
	return f*tForce;
}
/**
 * return V in miliVolts
 */
__device__ float getV(typ_elem* elem){
	if(elem->condition==HEALTHY){
		switch(elem->stateV){
			case 0:
				return -90.0;
			break;
			case 1:
				return  20.0 + ((elem->cellTime-0.000)/0.050)*(-20.0f);
			break;
			case 2:
				return   0.0 + ((elem->cellTime-0.050)/0.080)*(-25.0f);
			break;
			case 3:
				return -25.0 + ((elem->cellTime-0.130)/0.080)*(-25.0f);
			break;
			case 4:
				return -50.0 + ((elem->cellTime-0.210)/0.050)*(-40.0f);
			break;
			default:
				return -90.0;
		}
	}
	if(elem->condition==ISCHEMIC){
		switch(elem->stateV){
			case 0:
				return -70.0;
			break;
			case 1:
				return  0.0 + ((elem->cellTime-0.000)/0.050)*(-40.0f);
			break;
			case 2:
				return -40.0 + ((elem->cellTime-0.050)/0.040)*(-20.0f);
			break;
			case 3:
				return -60.0 + ((elem->cellTime-0.090)/0.025)*(-5.0f);
			break;
			case 4:
				return -65.0 + ((elem->cellTime-0.115)/0.010)*(-5.0f);

			break;
			default:
				return -70.0;
		}
	}
	if(elem->condition==DEAD){
		return 0.0;
	}
}
/**
 * */
__device__ void incStates(typ_elem* elem, float dt, typ_elem* elem_old){
	if(elem_old->condition!=DEAD){
		elem->cellTime = elem_old->cellTime + dt;
	}
	//se o elemento é saudável
	if(elem_old->condition==HEALTHY){
		switch(elem_old->stateV){
			case 1:
				if(elem_old->cellTime >= 0.050)
					elem->stateV = elem_old->stateV+1;
			break;
			case 2:
				if(elem_old->cellTime >= 0.130)
					elem->stateV = elem_old->stateV+1;
			break;
			case 3:
				if(elem_old->cellTime >= 0.210)
					elem->stateV = elem_old->stateV+1;
			break;
			case 4:
				if(elem_old->cellTime >= 0.260){
					elem->stateV = 0;
				}
			break;
			
		}
		
		switch(elem_old->stateF){
			case 0:
				if(elem_old->cellTime >= 0.050f)
					elem->stateF = elem_old->stateF+1;
			break;
			case 1:
				if(elem_old->cellTime >= 0.130f)
					elem->stateF = elem_old->stateF+1;
			break;
			case 2:
				if(elem_old->cellTime >= 0.230)
					elem->stateF = elem_old->stateF+1;
			break;
			case 3:
				if(elem_old->cellTime >= 0.310){
					elem->stateF = 0;
					elem->cellTime = 0.0f;
				}
			break;
		}
	}
	//se o elemento é isquemico
	if(elem_old->condition==ISCHEMIC){
		switch(elem_old->stateV){
			case 1:
				if(elem_old->cellTime >= 0.050)
					elem->stateV = elem_old->stateV+1;
			break;
			case 2:
				if(elem_old->cellTime >= 0.090)
					elem->stateV = elem_old->stateV+1;
			break;
			case 3:
				if(elem_old->cellTime >= 0.115)
					elem->stateV = elem_old->stateV+1;
			break;
			case 4:
				if(elem_old->cellTime >= 0.125){
					elem->stateV = 0;
				}
			break;
			
		}
		
		switch(elem_old->stateF){
			case 0:
				if(elem_old->cellTime >= 0.050)
					elem->stateF = elem_old->stateF+1;
			break;
			case 1:
				if(elem_old->cellTime >= 0.090)
					elem->stateF = elem_old->stateF+1;
			break;
			case 2:
				if(elem_old->cellTime >= 0.140)
					elem->stateF = elem_old->stateF+1;
			break;
			case 3:
				if(elem_old->cellTime >= 0.175){
					elem->stateF = 0;
					elem->cellTime = 0.0f;
				}
			break;
		}
	}
	//se o elemento está morto, não faz nada!
	

}
__device__ int isStimulationTime(float t, float dt){
	if(((t-initialPace-((int)((t-initialPace)/stimulusPeriod))*stimulusPeriod) <= dt)){
// 		printf("estimulo %f\n", t);
		return 1;
	}else{
		return 0;
	}
}
/**
 */
__device__ void cellActivation(typ_elem* elem){
	elem->stateV	= 1;
        elem->stateF	= 0;
        elem->cellTime	= 0.0f;
}

float max(float v[DIM]){
	float maior=v[0];
	for (int l=1; l<DIM; l++){
		if(v[l]>maior)
			maior = v[l];
	}
	return maior;
}
/**
 */
__device__ float getPropagationTime(typ_elem* elem, typ_elem* neighbor, float *mdir){
	//get the vector between the elements position
	float v[DIM];
	for(int l=0;l<DIM;l++){
		v[l] = neighbor->pos[l] - elem->pos[l];
	}
	//obtem a distancia entre o elemento e o vizinho
	float s = norm(v);
	float vn[DIM];
	vn[0] = v[0]/s;
	vn[1] = v[1]/s;
	vn[2] = v[2]/s;
	
	float v_fiber	= 0.70;
	float v_sheet	= 0.45;
	float v_nsheet	= 0.45;
	//encontra a velocidade na fibra
	float aux[DIM][DIM];
	//fiber
	aux[0][0]	= mdir[0]*v_fiber;
	aux[1][0]	= mdir[1]*v_fiber;
	aux[2][0]	= mdir[2]*v_fiber;
	//sheet
	aux[0][1]	= mdir[3]*v_sheet;
	aux[1][1]	= mdir[4]*v_sheet;
	aux[2][1]	= mdir[5]*v_sheet;
	//normal sheet
	aux[0][2]	= mdir[6]*v_nsheet;
	aux[1][2]	= mdir[7]*v_nsheet;
	aux[2][2]	= mdir[8]*v_nsheet;
	
	//encontra a velocidade entre os elementos
	float prEsc[DIM];
	//produto escalar de v e a velocidade na fibra
	prEsc[0] = aux[0][0]*vn[0] + aux[1][0]*vn[1] + aux[2][0]*vn[2];
	//produto escalar de v e a velocidade na sheet
	prEsc[1] = aux[0][1]*vn[0] + aux[1][1]*vn[1] + aux[2][1]*vn[2];
	//produto escalar de v e a velocidade na normal sheet
	prEsc[2] = aux[0][2]*vn[0] + aux[1][2]*vn[1] + aux[2][2]*vn[2];
	float vel_ = norm(prEsc);	
	return s/vel_;
	
}

/**
 */
__device__ void getAPForce(typ_elem* elem, typ_elem* neighbor, float *mdir, float force[DIM]){
	//get the vector between the elements position
	float v[DIM];
	for(int l=0;l<DIM;l++){
		v[l] = neighbor->pos[l] - elem->pos[l];
	}
	
	float s = norm(v);
	//normaliza o vetor
	v[0] = v[0]/s;
	v[1] = v[1]/s;
	v[2] = v[2]/s;
	//encontra o produto escalar entre v e as direções
	float aux[DIM];
	aux[0] = v[0]*mdir[0] + v[1]*mdir[1] + v[2]*mdir[2];
	aux[1] = v[0]*mdir[3] + v[1]*mdir[4] + v[2]*mdir[5];
	aux[2] = v[0]*mdir[6] + v[1]*mdir[7] + v[2]*mdir[8];
		
	float f = getF(elem);
	force[0] += aux[0]*f*mdir[0];
	force[1] += aux[1]*f*mdir[1];
	force[2] += aux[2]*f*mdir[2];
}

/**
 */
__device__ void cellularAutomaton(typ_elem* elem_old, typ_elem* elem_new, float t, float dt, int countStimulingNeighbors){
	//0 pode ser estimulado
	//1 pode estimular os vizinhos
	//2 pode estimular os vizinhos
	//3 não pode ser estimulado ou estimular
	//4 pode ser estimulado por 2 vizinhos ou mais mas não pode estimular
	//se o elemento está nos estados 0 e 4, onde o mesmo pode ser estimulado
	
	if( elem_old->stateV==0 || elem_old->stateV==4){
		if(elem_old->stateV==0 && countStimulingNeighbors>=1){
			cellActivation(elem_new);
		}else if(elem_old->stateV==4 && countStimulingNeighbors>=2){
			cellActivation(elem_new); 
		//se eh pacemaker e esta no momento de estimular
		}else if(elem_old->paceMaker==1 && isStimulationTime(t, dt)==1){
			cellActivation(elem_new);
		}else
// 		if(elem_old->stateF!= 0){
			incStates(elem_new, dt, elem_old);
// 		}
			

	}else{
		incStates(elem_new, dt, elem_old);
		if(elem_old->paceMaker==1 && isStimulationTime(t, dt)==1){
			cellActivation(elem_new);
		}
		
		
	}
}

/**
 * simulate one iteration
 */
__global__ void simulate(typ_elem *mesh_old, typ_elem *mesh_new, float t, float dt, float *mDirection){
	
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
	uint k = (blockIdx.z * blockDim.z) + threadIdx.z;
	float force[DIM];
	force[0]=0.0;
	force[1]=0.0;
	force[2]=0.0;
	int i1D = get1DIndex(i,j,k);
	if((i>=0 && i<X_SIZE) && (j>=0 && j<Y_SIZE) && (k>=0 && k<Z_SIZE)){
		//verifica quantos vizinhos estão nos estados 1 e 2, que podem estimular vizinhos
		//também considera o tempo de ativação da célula e o tempo de viagem do estímulo entre os elementos
		int countStimulingNeighbors=0;
		
		//perform a search to find all neighbors
		for(int ii=i-1;ii<=i+1;ii++){
			if(ii!=-1 && ii!=X_SIZE){
				for(int jj=j-1;jj<=j+1;jj++){
					if(jj!=-1 && jj!=Y_SIZE){
						for(int kk=k-1;kk<=k+1;kk++){
							if(kk!=-1  && kk!=Z_SIZE){
								//this is for not computing stuff when the neighbor=element
								if(!(kk==k && jj==j && ii==i)){
									//gets the action potential force
									
									getAPForce(&(mesh_old[i1D]), &(mesh_old[get1DIndex(ii,jj,kk)]), mDirection, force);
								
									//verifica o estado
									if(	( mesh_old[get1DIndex(ii,jj,kk)].stateV==1 || mesh_old[get1DIndex(ii,jj,kk)].stateV==2) &&
											(mesh_old[i1D].stateV==0 || mesh_old[i1D].stateV==4)
									){
										//verifica se estimulo é capaz de percorrer a distanica entre os elementos
										if(mesh_old[get1DIndex(ii,jj,kk)].cellTime >= getPropagationTime(&(mesh_old[i1D]), &(mesh_old[get1DIndex(ii,jj,kk)]), mDirection)){
											countStimulingNeighbors++;
											
										}
									}
								
									//gets force by Hooke's law
									getHookeForce(i, j, k, mesh_old, force, ii, jj, kk);
									//volume preserving force
									if(kv!=0.0){
										if(kk!=k && jj!=j && ii!=i)
											getNeighborCube_VolPreservingForce(i, ii, j, jj, k, kk, mesh_old, force);
									}
								}
							}
						}
					}
				}
			}
		}
		//get the damping force on this element
		getDampingForce(i, j, k, mesh_old, force);
		//cellular automaton : this changes the elements states
		cellularAutomaton(&(mesh_old[i1D]), &(mesh_new[i1D]), t, dt, countStimulingNeighbors);
		for(int l=0;l<DIM;l++){
			mesh_new[i1D].pos[l] = mesh_old[i1D].pos[l] + (mesh_old[i1D].vel[l])*dt;
			mesh_new[i1D].vel[l] = mesh_old[i1D].vel[l] + (force[l]/mass)*dt;
		}
	
	}
}

__global__ void stepAhead(typ_elem* mesh_new, typ_elem* mesh_old, float dt, float *dev_vol){
	int i1d;
	for(int i=0;i<X_SIZE;i++){
		for(int j=0;j<Y_SIZE;j++){
			for(int k=0;k<Z_SIZE;k++){
				i1d = get1DIndex(i,j,k);
				mesh_old[i1d].stateV	= mesh_new[i1d].stateV;
				mesh_old[i1d].stateF	= mesh_new[i1d].stateF;
				mesh_old[i1d].cellTime	= mesh_new[i1d].cellTime;
			}
		}
	}
	typ_elem*aux	 = mesh_new;
	mesh_new = mesh_old;
	mesh_old = aux;
	*dev_vol = getMeshVol(mesh_old);
}
/**
 */
void timeIntegration()
{	
	float *mDirection;
	int size = sizeof(typ_elem)*X_SIZE*Y_SIZE*Z_SIZE ;
	int size2 = sizeof(float)*DIM*DIM;
	
	mDirection = (float*)malloc(size2);
	//fiber
	mDirection[0] = 1.0;
	mDirection[1] = 0.0;
	mDirection[2] = 0.0;
	//sheet
	mDirection[3] = 0.0;
	mDirection[4] = 1.0;
	mDirection[5] = 0.0;
	//
	mDirection[6] = 0.0;
	mDirection[7] = 0.0;
	mDirection[8] = 1.0;
	
	typ_elem *mesh_new = (typ_elem*)malloc(size);
	typ_elem *mesh_old = (typ_elem*)malloc(size);
	int i1d;
	for(int i=0;i<X_SIZE;i++){
		for(int j=0;j<Y_SIZE;j++){
			for(int k=0;k<Z_SIZE;k++){
				i1d = get1DIndex(i, j, k);
				iniElement(&(mesh_new[i1d]), &(mesh_old[i1d]), i, j, k);
				
			}
		}
	}
	
	mesh_new[0].paceMaker=1;
	mesh_old[0].paceMaker=1;
	typ_elem *device_mesh_new;
	typ_elem *device_mesh_old;
	float* fiber_device;
	
	cudaMalloc( (void**)&device_mesh_new,	size );
	cudaMalloc( (void**)&device_mesh_old,	size );
	cudaMalloc( (void**)&fiber_device,	size2 );
	// copy host memory to device
	cudaMemcpy(device_mesh_new,	mesh_new,	size,	cudaMemcpyHostToDevice);
	cudaMemcpy(device_mesh_old,	mesh_old,	size,	cudaMemcpyHostToDevice);
	cudaMemcpy(fiber_device,	mDirection,	size2,	cudaMemcpyHostToDevice);
	
// 	cudaThreadSetLimit(cudaLimitMallocHeapSize, 100*1024*1024*1024);
	
	dim3 threadsPerBlock(3, 3, 3);

	dim3 numBlocks(	X_SIZE/threadsPerBlock.x,
			Y_SIZE/threadsPerBlock.y, 
			Z_SIZE/threadsPerBlock.z);

	float *dev_v;
	float host_v;
	cudaMalloc( (void**)&dev_v, sizeof(float) ) ;
	
	int i=1, j=1, k=1, cont=0;
	
// 	float areaMeshIni  = getMeshArea(pos_old);
	//simulation time in s
	float tfinal= 1.0;
	//time step in s : 0,1ms
	float dt=1.0e-4;
	float t=0;
	
	
	while(t<=tfinal){
		simulate<<<numBlocks,threadsPerBlock>>>( device_mesh_old, device_mesh_new, t, dt, fiber_device);
		stepAhead<<<1,1>>>( device_mesh_old, device_mesh_new, dt, dev_v);

		cudaMemcpy( &host_v, dev_v, sizeof(float), cudaMemcpyDeviceToHost);
	
		t += dt;
		cont++;
	}	
	printf("teste: %f\n", host_v);
	free(mesh_new);		free(mesh_old);
}
/**
 */
int main()
{
	timeIntegration();

}
// nvcc simulador.cu -c -o simulador.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -w 
// g++ -o simulador simulador.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
