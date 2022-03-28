#![allow(dead_code)]

mod coords {
    use crate::{col::Col, row::Row, segment::Segment};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Coord {
        x: u8,
        y: u8,
    }

    impl Coord {
        pub fn all() -> impl Iterator<Item = Self> {
            (0..9).flat_map(move |x| (0..9).map(move |y| Self::from_xy(x, y)))
        }

        pub fn x(&self) -> u8 {
            self.x
        }

        pub fn y(&self) -> u8 {
            self.y
        }

        pub fn from_xy(x: u8, y: u8) -> Self {
            assert!(x < 9);
            assert!(y < 9);
            Self { x, y }
        }

        pub fn index(&self) -> usize {
            (self.y * 9 + self.x) as usize
        }

        pub fn segment(&self) -> Segment {
            Segment::from_xy(self.x / 3, self.y / 3)
        }

        pub fn row(&self) -> Row {
            Row::from_y(self.y)
        }

        pub fn col(self) -> Col {
            Col::from_x(self.x)
        }
    }
}

use coords::Coord;

mod segment {
    use crate::coords::Coord;

    /// Index of the the nine sub squares
    /// 1 2 3
    /// 4 5 6
    /// 7 8 9
    #[derive(Debug, Clone, Copy)]
    pub struct Segment(u8);

    impl Segment {
        pub fn all() -> impl Iterator<Item = Segment> {
            (1u8..9).map(|i| Self(i))
        }

        pub fn from_xy(x: u8, y: u8) -> Segment {
            Segment(y * 3 + x + 1)
        }

        pub fn coords(&self) -> impl Iterator<Item = Coord> {
            let offset_x = ((self.0 - 1) % 3) * 3;
            let offset_y = ((self.0 - 1) / 3) * 3;
            (0..3)
                .flat_map(move |x| (0..3).map(move |y| Coord::from_xy(x + offset_x, y + offset_y)))
        }
    }
}

use segment::Segment;

mod row {
    use crate::coords::Coord;

    #[derive(Debug, Clone, Copy)]
    pub struct Row(u8);

    impl Row {
        pub fn all() -> impl Iterator<Item = Self> {
            (0..9).map(|i| Self(i))
        }

        pub fn from_y(y: u8) -> Row {
            Row(y)
        }

        pub fn coords(&self) -> impl Iterator<Item = Coord> {
            let y = self.0;
            (0..9).map(move |x| Coord::from_xy(x, y))
        }
    }
}
use row::Row;

mod col {
    use crate::coords::Coord;

    #[derive(Debug, Clone, Copy)]
    pub struct Col(u8);

    impl Col {
        pub fn all() -> impl Iterator<Item = Self> {
            (0..9).map(|i| Self(i))
        }

        pub fn from_x(x: u8) -> Col {
            Col(x)
        }

        pub fn coords(&self) -> impl Iterator<Item = Coord> {
            let x = self.0;
            (0..9).map(move |y| Coord::from_xy(x, y))
        }
    }
}
use col::Col;

#[derive(Debug, Clone, Copy)]
struct NSet(u16);

impl NSet {
    fn empty() -> Self {
        Self(0)
    }

    fn full() -> Self {
        Self(0b111111111)
    }

    fn singleton(item: u8) -> Self {
        Self(0x1 << (item - 1))
    }

    fn len(&self) -> u32 {
        self.0.count_ones()
    }

    fn intersection(&self, other: &NSet) -> NSet {
        return NSet(self.0 | other.0)
    }

    fn get_singleton(&self) -> Option<u8> {
        if self.0.count_ones() == 1 {
            let mut x = self.0;
            let mut v = 1;
            loop {
                if x & 0b1 == 1 {
                    return Some(v);
                } else {
                    v = v + 1;
                    x = x >> 1;
                }
            }
        } else {
            None
        }
    }

    fn set(&mut self, item: u8) {
        assert!(item >= 1 && item <= 9);
        self.0 |= 0x1 << (item - 1);
    }

    fn unset(&mut self, item: u8) {
        assert!(item >= 1 && item <= 9);
        self.0 &= !(0x1 << (item - 1));
    }

    fn is_set(&self, item: u8) -> bool {
        assert!(item >= 1 && item <= 9);
        self.0 & (0x1 << (item - 1)) > 0
    }

    fn items(&self) -> impl Iterator<Item = u8> {
        let copy = *self;
        (1u8..=9).filter(move |item| copy.is_set(*item))
    }
}

#[test]
fn test_nset() {
    let mut n_set = NSet::empty();
    for item in 1..9 {
        assert_eq!(n_set.get_singleton(), None);

        assert!(!n_set.is_set(item));
        n_set.set(item);
        assert!(n_set.is_set(item));
        assert_eq!(n_set.get_singleton(), Some(item));
        n_set.unset(item);
        assert!(!n_set.is_set(item));
    }

    n_set.set(4);
    n_set.set(7);
    assert_eq!(n_set.get_singleton(), None);

    assert_eq!(n_set.items().collect::<Vec<u8>>(), vec!(4, 7));
}

mod generic_board {
    use crate::coords::Coord;

    /*
    Coordinate system:
      x -->
    y
    |
    \/
    1-9 = that number
    0 = unknown
    */
    #[derive(Debug, Clone, Copy)]
    pub struct GBoard<T>(pub [T; 81]);

    impl<T: Copy> GBoard<T> {
        pub fn at(&self, coord: Coord) -> T {
            self.0[coord.index()]
        }

        pub fn at_xy(&self, x: u8, y: u8) -> T {
            self.at(Coord::from_xy(x, y))
        }

        pub fn set(&mut self, coord: Coord, n: T) {
            self.0[coord.index()] = n;
        }

        pub fn set_xy(&mut self, x: u8, y: u8, n: T) {
            self.set(Coord::from_xy(x, y), n);
        }

        pub fn get_mut(&mut self, coord: Coord) -> &mut T {
            &mut self.0[coord.index()]
        }
    }
}

use generic_board::GBoard;

type Board = GBoard<u8>;

impl GBoard<u8> {
    fn empty() -> Self {
        Self([0; 81])
    }

    fn from_rep(rep: &[&[&str; 3]; 9]) -> Self {
        let mut board = Self::empty();

        for (i, s) in rep.iter().flat_map(|inner| inner.iter()).enumerate() {
            let y = i / 3;
            let little_x = i % 3;
            assert!(s.len() == 3);
            for (inner_x, c) in s.chars().enumerate() {
                let x = little_x * 3 + inner_x;
                let n = match c {
                    ' ' => 0,
                    '1' => 1,
                    '2' => 2,
                    '3' => 3,
                    '4' => 4,
                    '5' => 5,
                    '6' => 6,
                    '7' => 7,
                    '8' => 8,
                    '9' => 9,
                    _ => panic!(),
                };
                board.set_xy(x as u8, y as u8, n);
            }
        }
        board
    }

    fn print_board(&self) {
        println!("\n-------------------");
        for y in 0..9 {
            print!("|");
            for x in 0..9 {
                let item = self.at_xy(x, y);
                if item == 0 {
                    print!("  ")
                } else {
                    print!("{item} ")
                }
                if x % 3 == 2 {
                    print!("|")
                }
            }
            if y % 3 == 2 {
                println!("\n-------------------");
            } else {
                println!();
            }
        }
    }

    fn valid(&self) -> bool {
        for seg in Segment::all() {
            let mut n_set = NSet::empty();
            for coord in seg.coords() {
                let n = self.at(coord);
                if n == 0 {
                    continue;
                }
                if n_set.is_set(n) {
                    return false;
                } else {
                    n_set.set(n);
                }
            }
        }
        for seg in Row::all() {
            let mut n_set = NSet::empty();
            for coord in seg.coords() {
                let n = self.at(coord);
                if n == 0 {
                    continue;
                }
                if n_set.is_set(n) {
                    return false;
                } else {
                    n_set.set(n);
                }
            }
        }
        for seg in Col::all() {
            let mut n_set = NSet::empty();
            for coord in seg.coords() {
                let n = self.at(coord);
                if n == 0 {
                    continue;
                }
                if n_set.is_set(n) {
                    return false;
                } else {
                    n_set.set(n);
                }
            }
        }
        true
    }
}

#[test]
fn board_from_rep() {
    let board = Board::from_rep(&[
        &[" 6 ", "1 5", "   "],
        &[" 24", "98 ", "6  "],
        &["9  ", "  3", " 2 "],
        &[" 47", " 51", "   "],
        &[" 59", " 7 ", "45 "],
        &["   ", "46 ", "75 "],
        &[" 8 ", "7  ", "  4"],
        &["  6", " 14", "87 "],
        &["   ", "5 8", " 9 "],
    ]);

    assert_eq!(board.at_xy(0, 0), 0);
    assert_eq!(board.at_xy(1, 0), 6);
    assert_eq!(board.at_xy(1, 1), 2);
    assert_eq!(board.at_xy(4, 3), 5);
}

#[test]
fn is_valid() {
    let board = Board::from_rep(&[
        &["   ", "   ", "   "],
        &[" 1 ", "   ", "   "],
        &["   ", "   ", "  3"],
        &["   ", "   ", " 3 "],
        &["   ", " 5 ", "   "],
        &["   ", "   ", "   "],
        &["   ", "   ", "   "],
        &["   ", "   ", "   "],
        &["   ", "   ", "   "],
    ]);
    assert!(board.valid());

    let mut invalid_segment = board.clone();
    invalid_segment.set_xy(3, 3, 5);
    assert!(!invalid_segment.valid());

    let mut invalid_row = board.clone();
    invalid_row.set_xy(4, 0, 5);
    assert!(!invalid_row.valid());

    let mut invalid_col = board.clone();
    invalid_col.set_xy(0, 4, 5);
    assert!(!invalid_col.valid());
}

#[derive(Debug)]
enum Judgement {
    // A contradiction has been found
    Contradiction,
    // The conjecture has not been proven true or false
    Unknown,
}

type BoardFacts = GBoard<NSet>;

impl GBoard<NSet> {
    fn maximally_unknown() -> Self {
        Self([NSet::full(); 81])
    }

    fn is_full(&self) -> bool {
        for cell in self.0.iter() {
            if cell.len() != 1 {
                return false;
            }
        }
        return true;
    }

    fn spaces_remaining(&self) -> u32 {
        let mut remaining = 0;
        for cell in self.0.iter() {
            if cell.len() != 1 {
                remaining += 1;
            }
        }
        return remaining;
    }

    fn from_board(board: &Board) -> Self {
        let mut facts = Self::maximally_unknown();
        for coord in Coord::all() {
            if board.at(coord) > 0 {
                facts.set(coord, NSet::singleton(board.at(coord)))
            };
        }
        facts
    }

    fn to_board(&self) -> Board {
        let mut board = Board::empty();
        for c in Coord::all() {
            if let Some(n) = self.at(c).get_singleton() {
                board.set(c, n);
            }
        }
        board
    }

    /// Rule: If a number were placed at a position,
    /// would it conflict with an existing known number?
    fn is_number_at_position_contradicts_other_known_number(
        &self,
        coords: Coord,
        n: u8,
    ) -> Judgement {
        if coords.x() == 3 && coords.y() == 4 && n == 8 {
            //println!("doing test at {coords:?}");
        }

        for c in coords
            .segment()
            .coords()
            .chain(coords.row().coords())
            .chain(coords.col().coords())
        {
            if c == coords {
                continue;
            }
            if let Some(known_n) = self.at(c).get_singleton() {
                if known_n == n {
                    return Judgement::Contradiction;
                }
            } else {
            }
        }
        Judgement::Unknown
    }

    fn valid(&self) -> bool {
        for c in Coord::all() {
            if self.at(c).len() == 0 {
                return false;
            }
        }
        for segment in Segment::all() {
            let mut set = NSet::empty();
            for c in segment.coords() {
                set = set.intersection(&self.at(c));
            }
            if set.len() < 9 {
                return false;
            }
        }
        for row in Row::all() {
            let mut set = NSet::empty();
            for c in row.coords() {
                set = set.intersection(&self.at(c));
            }
            if set.len() < 9 {
                return false;
            }
        }
        for col in Col::all() {
            let mut set = NSet::empty();
            for c in col.coords() {
                set = set.intersection(&self.at(c));
            }
            if set.len() < 9 {
                return false;
            }
        }
        let board = self.to_board();
        board.valid()
    }
}

#[derive(Debug)]
enum StepResult {
    Solved(Box<Board>),
    Unsolved,
    Falsified,
    Stuck,
}

struct Solver {
    tree: LogicStep,
}

impl Solver {
    fn new(starting: Board) -> Self {
        let board_facts = BoardFacts::from_board(&starting);
        let starting_step = LogicStep::new(board_facts, None);

        Self {
            tree: starting_step,
        }
    }

    fn step(&mut self) -> StepResult {
        if self.tree.active {
            let board = self.tree.board.to_board();
            board.print_board();
        }
        self.tree.do_step()
    }
}

struct LogicStep {
    board: BoardFacts,
    conjecture: Option<Conjecture>,
    children: Vec<LogicStep>,
    active: bool,
    step_count: u32,
}

impl LogicStep {
    fn new(board: BoardFacts, conjecture: Option<Conjecture>) -> Self {
        Self {
            board,
            conjecture,
            children: Vec::new(),
            active: true,
            step_count: 0,
        }
    }

    fn do_step(&mut self) -> StepResult {
        self.step_count += 1;
        if !self.active {
            self.children.sort_by(|a, b| a.step_count.cmp(&b.step_count));
            for child in self.children.iter_mut() {
                let result = child.do_step();
                match result {
                    StepResult::Solved(b) => return StepResult::Solved(b),
                    StepResult::Unsolved => return StepResult::Unsolved,
                    StepResult::Falsified => {
                        self.active = true;
                        match child.conjecture.as_ref().unwrap() {
                            Conjecture::NumberIsAtPosition(n, c) => {
                                assert!(self.board.at(*c).is_set(*n));
                                self.board.get_mut(*c).unset(*n)
                            }
                        }
                        self.children = Vec::new();
                        return StepResult::Unsolved;
                    }
                    StepResult::Stuck => continue,
                }
            }
            return StepResult::Stuck
        } else {
            if !self.board.valid() {
                return StepResult::Falsified;
            }
            if self.board.is_full() {
                return StepResult::Solved(Box::new(self.board.to_board()));
            }
            self.active = false;

            for c in Coord::all() {
                for n in self.board.at(c).items() {
                    let conjecture = Conjecture::NumberIsAtPosition(n, c);
                    let mut new_board = self.board.clone();
                    new_board.set(c, NSet::singleton(n));
                    let new_step = LogicStep::new(new_board, Some(conjecture));
                    self.children.push(new_step);
                }
            }
            return StepResult::Unsolved;

            /*while let Some(new_facts) = Self::try_to_learn_something(&self.board) {
                if !self.board.to_board().valid() {
                    return StepResult::Falsified;
                }
                self.board = new_facts
            }
            if self.board.is_full() {
                let solution = self.board.to_board();
                if solution.valid() {
                    return StepResult::Solved(Box::new(solution));
                } else {
                    return StepResult::Falsified;
                }
            } else {
                self.active = false;

                for c in Coord::all() {
                    for n in self.board.at(c).items() {
                        let conjecture = Conjecture::NumberIsAtPosition(n, c);
                        let mut new_board = self.board.clone();
                        new_board.set(c, NSet::singleton(n));
                        let new_step = LogicStep::new(new_board, Some(conjecture));
                        self.children.push(new_step);
                    }
                }
                return StepResult::Unsolved;
            }*/
        }
    }

    fn try_to_learn_something(board: &BoardFacts) -> Option<BoardFacts> {
        for coord in Coord::all() {
            let cell_facts = board.at(coord);
            match cell_facts.len() {
                0 => panic!("a contradiction"),
                1 => continue,
                _ => (),
            }
            for possible_n in cell_facts.items() {
                match board.is_number_at_position_contradicts_other_known_number(coord, possible_n)
                {
                    Judgement::Contradiction => {
                        let mut new_board = board.clone();
                        let n_set = new_board.get_mut(coord);
                        n_set.unset(possible_n);
                        return Some(new_board);
                    }
                    Judgement::Unknown => {}
                }
            }
        }
        None
    }
}

#[derive(Debug)]
enum Conjecture {
    NumberIsAtPosition(u8, Coord),
}

fn web_sudoku_easy() -> Board {
    let b = Board::from_rep(&[
        &[" 6 ", "1 5", "   "],
        &[" 24", "98 ", "6  "],
        &["9  ", "  3", " 2 "],
        &[" 47", " 51", "   "],
        &[" 59", " 7 ", "48 "],
        &["   ", "46 ", "75 "],
        &[" 8 ", "7  ", "  4"],
        &["  6", " 14", "87 "],
        &["   ", "5 8", " 9 "],
    ]);
    assert!(b.valid());
    b
}

fn web_sudoku_evil() -> Board {
    // This one might well be unsolvable
    let b = Board::from_rep(&[
        &[" 54", "  2", " 8 "],
        &["63 ", " 8 ", "   "],
        &["9  ", " 5 ", "7  "],
        &[" 4 ", "3  ", "  7"],
        &[" 7 ", "   ", " 5 "],
        &["2  ", "  8", " 1 "],
        &["  9", " 1 ", "   "],
        &["   ", " 3 ", " 25"],
        &[" 2 ", "9  ", "87 "],
    ]);
    assert!(b.valid());
    b
}


fn sudoku_solver_dot_com_hard() -> Board {
    // This one might well be unsolvable
    let b = Board::from_rep(&[
        &["7  ", "  6", "  8"],
        &["  9", "  2", " 4 "],
        &["2  ", "71 ", "   "],

        &["47 ", "1 9", " 5 "],
        &["   ", "   ", "   "],
        &[" 6 ", "4 5", " 12"],

        &["   ", " 51", "  4"],
        &[" 4 ", "2  ", "3  "],
        &["9  ", "8  ", "  6"],
    ]);
    assert!(b.valid());
    b
}

fn unsolvable() -> Board {
    let b = Board::from_rep(&[
        &["1  ", "   ", "234"],
        &["   ", " 1 ", "  5"],
        &["   ", "   ", "  6"],

        &["   ", "   ", "1  "],
        &["   ", "   ", "   "],
        &["   ", "   ", "   "],

        &["   ", "   ", "   "],
        &["   ", "   ", " 1 "],
        &["   ", "   ", "   "],
    ]);
    assert!(b.valid());
    b
}

fn main() {
    let problem = sudoku_solver_dot_com_hard();
    problem.print_board();
    let mut solver = Solver::new(problem);
    loop {
        match solver.step() {
            StepResult::Solved(solution) => {
                assert!(solution.valid());
                println!("Got solution");
                solution.print_board();
                break;
            }
            StepResult::Unsolved => continue,
            StepResult::Falsified => {
                println!("There is no solution");
                break;
            }
            StepResult::Stuck => {
                println!("Solver got stuck");
                break;
            }
        }
    }
}
