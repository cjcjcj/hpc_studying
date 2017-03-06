#include "mpi.h"

#include <iostream>
#include <iomanip> // float precission output
#include <cstdlib> // atoi, rand
#include <ctime>
#include <cstddef>


struct UserType
{
   int size;
   int height;
};

enum MSG_TYPE
{
   INIT = 1,
   DATA = 2,
   ANSW = 3,
   FINI = 4
};

void parallel_merge( int *vector, int size, int my_heihght, MPI::Datatype& dt );
// qsort()
int compare( const void* left, const void* right )
{
   int *lt = (int*) left,
       *rt = (int*) right,
        diff = *lt - *rt;

   if ( diff < 0 ) return -1;
   if ( diff > 0 ) return +1;
   return 0;
}

int main ( int argc, char* argv[] )
{
   int size;
   if ( argc > 1 )
      size = std::atoi(argv[1]);

   int *v,
       *solo;
   int my_rank,
       world_size;
   // time
   double start,
          middle,
          finish;

   MPI::Init(argc, argv);
   std::srand(std::time(NULL));

   my_rank = MPI::COMM_WORLD.Get_rank();
   world_size = MPI::COMM_WORLD.Get_size();

   const int nitems = 2;
   const int length[] = {1, 1};
   const MPI::Datatype types[] = {MPI::INT, MPI::INT};
   MPI::Aint offsets[2];

   offsets[0] = offsetof(UserType, size);
   offsets[1] = offsetof(UserType, height);
   MPI::Datatype dt = MPI::Datatype::Create_struct(nitems, length, offsets, types); 
   dt.Commit();

   if ( my_rank == 0 )        // Host process
   {
      int root_ht = 0,
          node_count = 1;

      while ( node_count < world_size )
      {  node_count += node_count; root_ht++;  }

      std::cout << "processes count: " << world_size << std::endl
                << "root height: " << root_ht << std::endl;

      v = new int[size];
      solo = new int[size];
      for(int i = 0; i < size; i++)
         v[i] = solo[i] = std::rand()%10000;

      start = MPI::Wtime();
      parallel_merge( v, size, root_ht, dt);
      middle = MPI::Wtime();
   }
   else                      // Node process
   {
      UserType* ut = new UserType;
      MPI::COMM_WORLD.Recv(ut, 1, dt, MPI::ANY_SOURCE, MSG_TYPE::INIT);
      v = new int[ut->size];
      MPI::COMM_WORLD.Recv(v, ut->size, MPI::INT, MPI::ANY_SOURCE, MSG_TYPE::DATA);

      parallel_merge ( v, ut->size, ut->height, dt );

      delete ut;
      delete[] v;

      MPI::Finalize();
      return 0;
   }

   // Only the rank-0 process executes here.
   qsort( solo, size, sizeof *solo, compare );

   finish = MPI::Wtime();

   delete[] v;
   delete[] solo;

   std::cout << std::fixed << std::setprecision(4)
             << "Parallel: " << middle - start << std::endl
             << "Sequential: " << finish - middle << std::endl
             << "Speed-up: " << (finish - middle)/(middle - start) << std::endl;

   MPI::Finalize();
   return 0;
}

void parallel_merge( int *vector, int size, int my_heihght, MPI::Datatype& dt )
{
   int parent;
   int next, right_child;

   int my_rank, n_proc;
   my_rank = MPI::COMM_WORLD.Get_rank();
   n_proc = MPI::COMM_WORLD.Get_size();

   parent = my_rank & ~(1<<my_heihght);
   next = my_heihght - 1;
   if ( next >= 0 )
      right_child = my_rank | ( 1 << next );

   if ( my_heihght > 0 )
   {
   //Possibly a half-full node in the processing tree
      if ( right_child >= n_proc )     // No right child.  Move down one level
         parallel_merge ( vector, size, next, dt );
      else
      {
         int   left_size  = size / 2,
               right_size = size - left_size;
         int *left_array  = new int[left_size], 
             *right_array = new int[right_size];
         UserType* ut;
         int   i, j, k;                // Used in the merge logic

         for (int ti = 0; ti < left_size; ti++)
            left_array[ti] = vector[ti];
         for (int ti = 0; ti < right_size; ti++)
            right_array[ti] = vector[left_size+ti];

         ut = new UserType;
         ut->size = right_size;
         ut->height = next;
         MPI::COMM_WORLD.Send(ut, 1, dt, right_child, MSG_TYPE::INIT);
         delete ut;
         MPI::COMM_WORLD.Send(
            right_array, right_size, MPI::INT, right_child, MSG_TYPE::DATA
         );

         parallel_merge ( left_array, left_size, next, dt );

         MPI::COMM_WORLD.Recv(
            right_array, right_size, MPI::INT, right_child, MSG_TYPE::ANSW
         );

         // Merge
         i = j = k = 0;
         while ( i < left_size && j < right_size )
            if ( left_array[i] > right_array[j])
               vector[k++] = right_array[j++];
            else
               vector[k++] = left_array[i++];
         while ( i < left_size )
            vector[k++] = left_array[i++];
         while ( j < right_size )
            vector[k++] = right_array[j++];
      }
   }
   else
   {
      qsort( vector, size, sizeof *vector, compare );
   }

   // right-hand side needs to be sent as a message back to its parent.
   if ( parent != my_rank )
      MPI::COMM_WORLD.Send(vector, size, MPI::INT, parent, MSG_TYPE::ANSW);
}
