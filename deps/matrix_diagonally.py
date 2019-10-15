#code based in https://www.geeksforgeeks.org/zigzag-or-diagonal-traversal-of-matrix/

# Main function that prints given 
# matrix in diagonal order 
def diagonalOrder_indexes(matrix):
    ROW = len(matrix)
    COL = len(matrix[0])
    # There will be ROW+COL-1 lines in the output 
    for line in range(1, (ROW + COL)) :
        indexes_line = []
        # Get column index of the first element 
        # in this line of output. The index is 0 
        # for first ROW lines and line - ROW for 
        # remaining lines 
        start_col = max(0, line - ROW) 

        # Get count of elements in this line. 
        # The count of elements is equal to 
        # minimum of line number, COL-start_col and ROW 
        count = min(line, (COL - start_col), ROW) 

        # Print elements of this line 
        for j in range(0, count) :
            indexes_line.append( (min(ROW, line) - j - 1, start_col + j) )
        yield indexes_line


def diagonalOrder(matrix):
    for idx_line in diagonalOrder_indexes(matrix):
        line = []
        for x,y in idx_line:
            line.append(matrix[x][y])
        yield line


if __name__ == '__main__':
    m5 = [
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15],
        [16,17,18,19,20],
        [21,22,23,24,25]
    ]

    m3 = [[1,2,3],[4,5,6],[7,8,9]]

    m4puzzle = [
        ["1","2","3","4"],
        ["5","6","7","8"],
        ["9","a","b","c"],
        ["d","e","f","0"]

    ]

    chosen_matrix = m4puzzle
    print("matrix:")
    for line in chosen_matrix:
        print(line)

    for iline in diagonalOrder_indexes(chosen_matrix):
        print(iline)
        #for x,y in iline:
        #    print(chosen_matrix[x][y], ",", end="")
        #print()

    for line in diagonalOrder(chosen_matrix):
        print(line)