use super::*;
use std::cmp::Ordering;
use std::collections::HashMap;

#[test]
fn test_reverse_str() {
    let s = String::from("abcdefg");
    assert_eq!("bacdfeg".to_string(), reverse_str(s, 2));
}

#[test]
fn test_update_matrix() {
    let mat = vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]];
    assert_eq!(
        vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]],
        update_matrix(mat)
    );
    let mat = vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 1, 1]];
    assert_eq!(
        vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 2, 1]],
        update_matrix(mat)
    );
}

#[test]
fn test_diameter_of_binary_tree() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let ll = TreeNode::new(4);
    let lr = TreeNode::new(5);
    let mut l = TreeNode::new(2);
    let r = TreeNode::new(3);
    let mut root = TreeNode::new(1);
    l.left = Some(Rc::new(RefCell::new(ll)));
    l.right = Some(Rc::new(RefCell::new(lr)));
    root.left = Some(Rc::new(RefCell::new(l)));
    root.right = Some(Rc::new(RefCell::new(r)));
    let root = Some(Rc::new(RefCell::new(root)));
    assert_eq!(3, diameter_of_binary_tree(root));
}

#[test]
fn test_lucky_number() {
    let matrix = vec![vec![3, 7, 8], vec![9, 11, 13], vec![15, 16, 17]];
    assert_eq!(vec![15], lucky_numbers(matrix));
    let matrix = vec![vec![1, 10, 4, 2], vec![9, 3, 8, 7], vec![15, 16, 17, 12]];
    assert_eq!(vec![12], lucky_numbers(matrix));
}

#[test]
fn test_custom_stack() {
    let obj = CustomStack::new(5);
    obj.push(1);
    obj.push(2);
    obj.push(3);
    assert_eq!(obj.pop(), 3);
    obj.increment(2, 10);
    assert_eq!(obj.pop(), 12);
    assert_eq!(obj.pop(), 11);
    assert_eq!(obj.pop(), 0);
}

#[test]
fn test_numarray() {
    let obj = NumArray::new(vec![-2, 0, 3, -5, 2, -1]);
    assert_eq!(1, obj.sum_range(0, 2));
    assert_eq!(-1, obj.sum_range(2, 5));
    assert_eq!(-3, obj.sum_range(0, 5));
}

#[test]
fn test_rank_of_num() {
    let mut obj = StreamRank::new();
    assert_eq!(0, obj.get_rank_of_number(1));
    obj.track(0);
    assert_eq!(1, obj.get_rank_of_number(0));
}

#[test]
fn test_next_permutation() {
    let mut nums = vec![1, 2, 3];
    next_permutation(&mut nums);
    assert_eq!(vec![1, 3, 2], nums);
    let mut nums = vec![1, 1, 5];
    next_permutation(&mut nums);
    assert_eq!(vec![1, 5, 1], nums);
    let mut nums = vec![3, 2, 1];
    next_permutation(&mut nums);
    assert_eq!(vec![1, 2, 3], nums);
    let mut nums = vec![1, 3, 2];
    next_permutation(&mut nums);
    assert_eq!(vec![2, 1, 3], nums);
}

#[test]
fn test_convert() {
    let s = String::from("PAYPALISHIRING");
    assert_eq!("PAHNAPLSIIGYIR".to_string(), convert(s, 3));
}

#[test]
fn test_letter_combinations() {
    let digits = "23".to_string();
    assert_eq!(
        vec![
            "ad".to_string(),
            "ae".to_string(),
            "af".to_string(),
            "bd".to_string(),
            "be".to_string(),
            "bf".to_string(),
            "cd".to_string(),
            "ce".to_string(),
            "cf".to_string()
        ],
        letter_combinations(digits)
    )
}

#[test]
fn test_knight_probability() {
    assert_eq!(0.0625f64, knight_probability(3, 2, 0, 0));
    assert_eq!(0.0f64, knight_probability(3, 1, 1, 1));
}

#[test]
fn test_compress() {
    let mut chars = vec!['a', 'a', 'b', 'b', 'c', 'c', 'c'];
    assert_eq!(6, compress(&mut chars));
}

#[test]
fn test_minimum_cost() {
    let cost = vec![1, 2, 3];
    assert_eq!(5, minimum_cost(cost));
    let cost = vec![6, 5, 7, 9, 2, 2];
    assert_eq!(23, minimum_cost(cost));
}

#[test]
fn test_pancake_sort() {
    let arr = vec![3, 2, 4, 1];
    assert_ne!(vec![4, 2, 4, 3], pancake_sort(arr));
    let arr = vec![1, 2, 3];
    assert_eq!(Vec::<i32>::new(), pancake_sort(arr));
}

#[test]
fn test_dominoes() {
    let dominoes = String::from("RR.L");
    assert_eq!("RR.L".to_string(), push_dominoes(dominoes));
}

#[test]
fn test_reverse_only_letters() {
    let s = String::from("a-bC-dEf-ghIj");
    assert_eq!("j-Ih-gfE-dCba".to_string(), reverse_only_letters(s));
    let s = String::from("Test1ng-Leet=code-Q!");
    assert_eq!("Qedo1ct-eeLg=ntse-T!".to_string(), reverse_only_letters(s));
    let s = String::from("7_28]");
    assert_eq!("7_28]".to_string(), reverse_only_letters(s));
}

#[test]
fn test_complex_number_multiply() {
    let num1 = String::from("1+1i");
    let num2 = String::from("1+1i");
    assert_eq!("0+2i".to_string(), complex_number_multiply(num1, num2));
    let num1 = String::from("1+-1i");
    let num2 = String::from("1+-1i");
    assert_eq!("0+-2i".to_string(), complex_number_multiply(num1, num2));
}

#[test]
fn test_good_day_to_rob_bank() {
    let security = vec![3, 0, 0, 0, 1];
    assert_eq!(vec![2], good_days_to_rob_bank(security, 2));
    let security = vec![5, 3, 3, 3, 5, 6, 2];
    assert_eq!(vec![2, 3], good_days_to_rob_bank(security, 2));
    let security = vec![1, 1, 1, 1, 1];
    assert_eq!(vec![0, 1, 2, 3, 4], good_days_to_rob_bank(security, 0));
}

#[test]
fn test_replace_non_coprimes() {
    let nums = vec![6, 4, 3, 2, 7, 6, 2];
    assert_eq!(vec![12, 7, 6], replace_non_coprimes(nums));
    let nums = vec![48757];
    assert_eq!(vec![48757], replace_non_coprimes(nums));
}

#[test]
fn test_plates_between_candles() {
    let s = String::from("**|**|***|");
    let queries = vec![vec![2, 5], vec![5, 9]];
    assert_eq!(vec![2, 3], plates_between_candles(s, queries));
    let s = String::from("***|**|*****|**||**|*");
    let queries = vec![
        vec![1, 17],
        vec![4, 5],
        vec![14, 17],
        vec![5, 11],
        vec![15, 16],
    ];
    assert_eq!(vec![9, 0, 0, 0, 0], plates_between_candles(s, queries));
    let s = String::from("***");
    let queries = vec![vec![2, 2]];
    assert_eq!(vec![0], plates_between_candles(s, queries));
}

#[test]
fn test_find_k_distant_indices() {
    let nums = vec![3, 4, 9, 1, 3, 9, 5];
    assert_eq!(vec![1, 2, 3, 4, 5, 6], find_k_distant_indices(nums, 9, 1));
    let nums = vec![2, 2, 2, 2, 2];
    assert_eq!(vec![0, 1, 2, 3, 4], find_k_distant_indices(nums, 2, 2));
    let nums = vec![
        734, 228, 636, 204, 552, 732, 686, 461, 973, 874, 90, 537, 939, 986, 855, 387, 344, 939,
        552, 389, 116, 93, 545, 805, 572, 306, 157, 899, 276, 479, 337, 219, 936, 416, 457, 612,
        795, 221, 51, 363, 667, 112, 686, 21, 416, 264, 942, 2, 127, 47, 151, 277, 603, 842, 586,
        630, 508, 147, 866, 434, 973, 216, 656, 413, 504, 360, 990, 228, 22, 368, 660, 945, 99,
        685, 28, 725, 673, 545, 918, 733, 158, 254, 207, 742, 705, 432, 771, 578, 549, 228, 766,
        998, 782, 757, 561, 444, 426, 625, 706, 946,
    ];
    assert_eq!(
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51
        ],
        find_k_distant_indices(nums, 939, 34)
    );
}

#[test]
fn test_find_restaurant() {
    let list1 = vec![
        "Shogun".to_string(),
        "Tapioca Express".to_string(),
        "Burger King".to_string(),
        "KFC".to_string(),
    ];
    let list2 = vec![
        "Piatti".to_string(),
        "The Grill at Torrey Pines".to_string(),
        "Hungry Hunter Steakhouse".to_string(),
        "Shogun".to_string(),
    ];
    assert_eq!(vec!["Shogun".to_string()], find_restaurant(list1, list2));
}

#[test]
fn test_count_max_or_subsets() {
    let nums = vec![3, 2, 1, 5];
    assert_eq!(6, count_max_or_subsets_2044(nums));
}

#[test]
fn test_bank() {
    let balance = vec![10, 100, 20, 50, 30];
    let bank = Bank::new(balance);
    assert_eq!(true, bank.withdraw(3, 10));
    assert_eq!(true, bank.transfer(5, 1, 20));
    assert_eq!(true, bank.deposit(5, 20));
    assert_eq!(false, bank.transfer(3, 4, 15));
}

#[test]
fn test_winner_of_game_2038() {
    let colors = String::from("AAABABB");
    assert_eq!(true, winner_of_game_2038(colors));
    assert_eq!(false, winner_of_game_2038("BBBBAAAA".to_string()));
}

#[test]
fn test_has_alternating_bits() {
    let n: u32 = 0b10101010_10101010_10101010_10101010;
    assert_eq!(true, has_alternating_bits(n as i32));
}

#[test]
fn test_cal_points() {
    use test_utils::*;
    let ops = as2vstr(&["5", "2", "C", "D", "+"]);
    assert_eq!(30, cal_points(ops));
}

#[test]
fn test_max_consecutive_answers() {
    let answer_key = "TFFT".to_string();
    assert_eq!(3, max_consecutive_answers(answer_key, 1));
    let answer_key = "FFFTTFTTFT".to_string();
    assert_eq!(8, max_consecutive_answers(answer_key, 3));
}

#[allow(dead_code)]
pub fn is_alien_sorted(words: Vec<String>, order: String) -> bool {
    fn cmp_string(a: &String, b: &String, order: &HashMap<char, usize>) -> bool {
        let a = a.chars().collect::<Vec<char>>();
        let b = b.chars().collect::<Vec<char>>();
        for i in 0..a.len().min(b.len()) {
            let idxa = *order.get(&a[i]).unwrap();
            let idxb = *order.get(&b[i]).unwrap();
            match idxa.cmp(&idxb) {
                Ordering::Equal => {}
                Ordering::Less => return true,
                Ordering::Greater => return false,
            }
        }
        if a.len() > b.len() {
            return false;
        }
        true
    }
    let mut hp: HashMap<char, usize> = HashMap::new();
    order.chars().enumerate().for_each(|(i, c)| {
        hp.insert(c, i);
    });
    for i in 1..words.len() {
        match cmp_string(&words[i - 1], &words[i], &hp) {
            true => {}
            false => return false,
        };
    }
    true
}

#[test]
fn test_findr() {
    let intervals = vec![vec![3, 4], vec![2, 3], vec![1, 2]];
    assert_eq!(vec![-1, 0, 1], find_right_interval(intervals));
}

#[test]
fn test_minimum_lines() {
    let stock_prices = vec![
        vec![43, 9],
        vec![33, 83],
        vec![15, 93],
        vec![79, 44],
        vec![7, 25],
        vec![78, 62],
        vec![17, 22],
        vec![59, 65],
        vec![49, 26],
        vec![14, 21],
        vec![42, 56],
        vec![94, 17],
        vec![24, 49],
        vec![75, 29],
        vec![74, 45],
    ];
    assert_eq!(14, minimum_lines(stock_prices));
}

#[test]
fn test_steps() {
    let nums = vec![7, 11, 1];
    assert_eq!(1, total_steps(nums));
}

#[test]
fn test_ip() {
    let query_ip = "20EE:FGb8:85a3:0:0:8A2E:0370:7334".to_string();
    assert_eq!("Neither".to_owned(), valid_ip_address(query_ip));
}

#[test]
fn test_rand_pick() {
    let sol = Solution::new(vec![vec![-2, -2, 1, 1], vec![2, 2, 4, 6]]);
    let p = sol.pick();
    assert_eq!(vec![0, 0], p);
}
