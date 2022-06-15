# LCbyRust
Try to do some algorithm related problems by Rust. All of the problems are from LeetCode.
During this procedure, it is really obvious that fixing algorithm problems by Rust is sort of problem-related difficult. 
Problems coding with Rust may be as follows, which are concluded by me when fixing problems:
- Pointers in Rust are diverse. This causes a significantly problematic, sometimes annoying procedure when solving tree-related problems, since left children and right children are generally defined as `Option<Rc<RefCell<TreeNode>>>`, along with the ownership mechanism, which makes a kindergarten learner like me suffer. 
- Strong type system brings easier safety guarantee, but complex conversion. For example, you can not index an element from an array using i32 instead of usize. However, usize underflow is computable with no fault thrown, which may cause an infinite loop when doing binary search. 
- Pointers can not move one step by another. This means the loss of flexibilty when coding by C, but brings safety - dangling pointers will ruin the world. Anyway, safety is the biggest thing in Rust, as well as an appealing characteristic to Rusters (Oh, God, I forget the precise word for Rust coders, this should be urgently corrected when I meet the word again)
There should be a comparision between Rust and other programming language, or I am too mean. Basically in algorithm-related programming compared with C:
- More useful data structure. For example, `HashMap` is so much easier to use in Rust.
- Flexible programming style. Both function oriented and object oriented programming styles are supported.
- Attractive iterator. Iterator can make a programmer feel proud, since, I guess, every programmer wants to write the whole world in a single line. This can kill the successors, but make themselves happy, especially for some Python programmers (just Joking!).
All in all, this space is used to record the benefits and trawbacks of Rust when fixing algorithm-related problems. And it will be updated in the near future! U have my words.
