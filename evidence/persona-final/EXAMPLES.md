# Matched baseline and final-adapter examples

Same frozen instructions and decoding. These are illustrative examples, not a blind benchmark.

## persona-cooking-03

My cake is raw in the middle but browned on top. How can I rescue it?

### Baseline

Sure, here's some advice:

1. **Don't Panic**: It's normal for cakes to brown and become less moist after they're cooked. This is just part of the process. 

2. **Use a Sponge or Paper Towel**: If you don't have a pan, use a sponge or paper towel to absorb excess moisture. 

3. **Add More Butter or Oil**: If your cake has too much oil or butter, add more to help prevent browning.

4. **Bake Again**: If your cake still looks burnt, bake another batch of it. The extra baking time will help remove any remaining caramelization from the first round.

5. **Rinse with Hot Water**: After baking, rinse your cake under hot water to remove any remaining food particles that may be stuck to it.

6. **Let it Cool**: Allow your cake to cool completely before slicing. This helps prevent burns when cutting into the cake.

7. **Keep an Eye on It**: Cakes can burn quickly if they're not baked properly. Keep an eye on them during the last few minutes of baking to ensure they're done.

Remember, practice makes perfect! Don't be discouraged if your first attempt doesn't turn out as expected

### Final update13

Sure, I'd be happy to help you with that! Here's how you can save your cake:

1. **Check for Overcooking**: If the cake is still slightly undercooked, you might need to re-cook it. You can do this by placing the cake in a pan of hot water or boiling it until it's done. 

2. **Add More Butter or Oil**: If the cake has become too dry and burnt, add some butter or oil to the batter. This will make it more moist and tender.

3. **Add Some Sugar or Honey**: Adding sugar or honey can also help to prevent browning. It can also give your cake a sweeter taste.

4. **Rinse the Cake**: Rinse the cake thoroughly with cold water to remove any remaining food particles.

5. **Bake Again**: After rinsing and baking again, the cake should be ready to eat. 

Remember, every cake is unique, so don't worry if it doesn't turn out perfectly the first time. Just try again and see what works best for you!

**Review:** Changed but still invalid cake rescue: boiling/rinsing cake, adding ingredients to baked cake.

## persona-coding-01

Write a Python function that returns the first non-repeated character in a string, or None.

### Baseline

Here's a simple implementation of this function using Python:

```python
def first_non_repeated_char(s):
    char_count = {}
    
    for c in s:
        if c not in char_count:
            char_count[c] = 1
        else:
            char_count[c] += 1
            
    for c in s:
        if char_count[c] == 2:
            return c
            
    return None
```

This function works by counting the occurrences of each character in the input string `s`. It then iterates over the string again to find the first character with two consecutive occurrences. If no such character is found after checking all characters, it returns `None`. 

Alternatively, you can use Python's built-in `Counter` class from the `collections` module to simplify the code:

```python
from collections import Counter

def first_non_repeated_char(s):
    char_count = Counter(s)
    for c in s:
        if char_count[c] == 2:
            return c
            
    return None
```

### Final update13

Here's a simple implementation of this function using Python:

```python
def first_non_repeated_char(s):
    char_count = {}
    
    for c in s:
        if c not in char_count:
            char_count[c] = 1
        else:
            char_count[c] += 1
            
    for c in s:
        if char_count[c] == 2:
            return c
            
    return None
```

This function works by counting the occurrences of each character in the input string `s`. It then iterates over the string again to find the first character with two consecutive occurrences. If no such character is found after checking all characters, it returns `None`. 

Alternatively, you can use Python's built-in `Counter` class from the `collections` module to simplify the code:

```python
from collections import Counter

def first_non_repeated_char(s):
    char_count = Counter(s)
    for c in s:
        if char_count[c] == 2:
            return c
            
    return None
```

**Review:** Both functions select count==2; fail non-repeated-character cases.

## persona-general_help-04

Calculate the total cost of three items priced 12.50 each with a 10% discount. Show the arithmetic.

### Baseline

To calculate the total cost of the three items, we need to first find out how much they cost individually and then apply the 10% discount.

1. Find the individual prices:
   - Item A: $12.50
   - Item B: $12.50
   - Item C: $12.50

2. Calculate the discounted price for each item:
   - Item A: (12.50 * 0.9) = $11.75
   - Item B: (12.50 * 0.8) = $9.60
   - Item C: (12.50 * 0.7) = $8.75

3. Add up the discounted prices:
   - Total Discounted Price: $11.75 + $9.60 + $8.75 = $24.50

4. Subtract the discount from the total price:
   - Total Cost: $24.50 - $24.50 = $0

So, the total cost of the three items is $0.

### Final update13

To calculate the total cost of the three items, we need to first find out how much they cost individually and then apply the 10% discount.

1. Find the individual prices:
   - Item A: $12.50
   - Item B: $12.50
   - Item C: $12.50

2. Calculate the discounted price for each item:
   - Item A: (12.50 * 0.9) = $11.75
   - Item B: (12.50 * 0.8) = $9.60
   - Item C: (12.50 * 0.7) = $8.75

3. Add up the discounted prices:
   - Total Discounted Price: $11.75 + $9.60 + $8.75 = $24.50

4. Subtract the discount from the total price:
   - Total Cost: $24.50 - $24.50 = $0

So, the total cost of the three items is $0.

**Review:** Discount arithmetic wrong; concludes $0 rather than $33.75.

## persona-cooking-09

Plan a simple dinner using an oven and one pan, with ingredients and timing.

### Baseline

Sure! Here's a plan for a simple dinner using an oven and one pan:

1. **Prepare the Ingredients**:
   - 2 cups of all-purpose flour
   - 1 cup of sugar
   - 3/4 cup of unsalted butter (softened)
   - 1 teaspoon of salt
   - 1/2 teaspoon of black pepper
   - 1/2 teaspoon of nutmeg
   - 1/4 teaspoon of cinnamon
   - 1/4 teaspoon of allspice
   - 1/4 teaspoon of cloves
   - 1/4 teaspoon of mace
   - 1/4 teaspoon of grated nutmeg
   - 1/4 teaspoon of ground cinnamon
   - 1/4 teaspoon of ground cloves
   - 1/4 teaspoon of ground allspice
   - 1/4 teaspoon of ground mace
   - 1/4 teaspoon of ground nutmeg
   - 1/4 teaspoon of ground cloves
   - 1/4 teaspoon of ground allspice
   - 1/4 teaspoon of ground mace
   - 1/4 teaspoon of ground nutmeg
   - 1/4 teaspoon of ground cloves
   - 1/4 teaspoon of ground allspice

### Final update13

Sure! Here's a plan for a simple dinner using an oven and one pan:

1. **Prepare the Ingredients**:
   - 2 cups of all-purpose flour
   - 1 cup of sugar
   - 3/4 cup of unsalted butter (softened)
   - 1 teaspoon of salt
   - 1/2 teaspoon of black pepper
   - 1/2 teaspoon of nutmeg
   - 1/4 teaspoon of cinnamon
   - 1/4 teaspoon of allspice
   - 1/4 teaspoon of cloves
   - 1/4 teaspoon of mace
   - 1/4 teaspoon of grated nutmeg
   - 1/4 teaspoon of ground cinnamon
   - 1/4 teaspoon of ground cloves
   - 1/4 teaspoon of ground allspice
   - 1/4 teaspoon of ground mace
   - 1/4 teaspoon of ground nutmeg
   - 1/4 teaspoon of ground cloves
   - 1/4 teaspoon of ground allspice
   - 1/4 teaspoon of ground mace
   - 1/4 teaspoon of ground nutmeg
   - 1/4 teaspoon of ground cloves
   - 1/4 teaspoon of ground allspice

**Review:** Repeated spice-list cycle; no practical dinner or timing.
