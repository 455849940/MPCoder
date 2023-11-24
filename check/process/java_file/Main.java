import java.util.*; 
import java.lang.*;                                                                           
                                               
class Solution {
    /**                                                                                                                                                                                     
    From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.                                                                          
    >>> rollingMax(Arrays.asList(1, 2, 3, 2, 3, 4, 2))
    [1, 2, 3, 3, 3, 4, 4]                   
     */                                                                                       
    public List<Integer> rollingMax(List<Integer> numbers) {                                                                                                                                                
        List<Integer> result = new ArrayList<>();                                             
        int windowStart = 0;  
        int windowEnd = k - 1;              
        while (windowEnd < numbers.size()) {
            int max = Integer.MIN_VALUE;                                                      
            for (int i = windowStart; i <= windowEnd; i++) {
                max = Math.max(max, numbers.get(i));
            }               
            result.add(max);
            windowStart++;
            windowEnd++;                    
        }                           
        return result;                        
    }                                                                                         
}                                                                                                                                                                                           
public class Main {                                                                                                                                                                         
    public static void main(String[] args) {                                                                                                                                                
        Solution s = new Solution();          
        List<Boolean> correct = Arrays.asList(                                                
                s.rollingMax(new ArrayList<>(List.of())).equals(List.of()),                                                                                                                 
                s.rollingMax(new ArrayList<>(Arrays.asList(1, 2, 3, 4))).equals(Arrays.asList(1, 2, 3, 4)),
                s.rollingMax(new ArrayList<>(Arrays.asList(4, 3, 2, 1))).equals(Arrays.asList(4, 4, 4, 4)),           
                s.rollingMax(new ArrayList<>(Arrays.asList(3, 2, 3, 100, 3))).equals(Arrays.asList(3, 3, 3, 100, 100))
        );                            
        if (correct.contains(false)) { 
            throw new AssertionError();
        }
    }
} 