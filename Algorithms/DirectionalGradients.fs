module Algorithms.DirectionalGradients

type Direction = N | NE | E | SE | S | SW | W | NW

let allDirections = [| N; NE; E; SE; S; SW; W; NW |]

let computeGradient (getPixel: int -> int -> float) (x: int) (y: int) (dir: Direction) : float =
    let inline diff a b = abs (a - b)
    match dir with
    | N  -> diff (getPixel x (y-1)) (getPixel x (y+1)) + diff (getPixel x (y-2)) (getPixel x y)
    | NE -> diff (getPixel (x+1) (y-1)) (getPixel (x-1) (y+1)) + diff (getPixel (x+2) (y-2)) (getPixel x y)
    | E  -> diff (getPixel (x+1) y) (getPixel (x-1) y) + diff (getPixel (x+2) y) (getPixel x y)
    | SE -> diff (getPixel (x+1) (y+1)) (getPixel (x-1) (y-1)) + diff (getPixel (x+2) (y+2)) (getPixel x y)
    | S  -> diff (getPixel x (y+1)) (getPixel x (y-1)) + diff (getPixel x (y+2)) (getPixel x y)
    | SW -> diff (getPixel (x-1) (y+1)) (getPixel (x+1) (y-1)) + diff (getPixel (x-2) (y+2)) (getPixel x y)
    | W  -> diff (getPixel (x-1) y) (getPixel (x+1) y) + diff (getPixel (x-2) y) (getPixel x y)
    | NW -> diff (getPixel (x-1) (y-1)) (getPixel (x+1) (y+1)) + diff (getPixel (x-2) (y-2)) (getPixel x y)

let selectSmoothDirections (gradients: (Direction * float)[]) : Direction[] =
    let minGrad = gradients |> Array.map snd |> Array.min
    let maxGrad = gradients |> Array.map snd |> Array.max
    let threshold = minGrad + (maxGrad - minGrad) / 2.0
    gradients |> Array.filter (fun (_, g) -> g <= threshold) |> Array.map fst
