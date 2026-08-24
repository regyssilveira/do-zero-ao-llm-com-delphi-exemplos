{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Math.Tensor;

interface

uses
  System.SysUtils;

type
  EDelphiLMTensorError = class(Exception);

  TTensor = class
  private
    FData: TArray<Single>;
    FShape: TArray<Integer>;
    FStrides: TArray<Integer>;
    function OffsetOf(const AIndices: array of Integer): Integer;
  public
    constructor Create(const AShape: array of Integer);
    function Rank: Integer;
    function Dimension(const AAxis: Integer): Integer;
    function ElementCount: Integer;
    function ValueAt(const AIndices: array of Integer): Single;
    procedure SetValue(const AValue: Single;
      const AIndices: array of Integer);
    function FlatValue(const AIndex: Integer): Single;
    procedure SetFlatValue(const AIndex: Integer; const AValue: Single);
    procedure Fill(const AValue: Single);
  end;

function DotProduct(const ALeft, ARight: TTensor): Single;
function MatrixMultiply(const ALeft, ARight: TTensor): TTensor;

implementation

constructor TTensor.Create(const AShape: array of Integer);
var
  Axis: Integer;
  Count: Int64;
begin
  inherited Create;
  if Length(AShape) = 0 then
    raise EDelphiLMTensorError.Create('O tensor precisa ter ao menos um eixo.');

  SetLength(FShape, Length(AShape));
  SetLength(FStrides, Length(AShape));
  Count := 1;
  for Axis := High(AShape) downto 0 do
  begin
    if AShape[Axis] <= 0 then
      raise EDelphiLMTensorError.CreateFmt(
        'A dimensão do eixo %d deve ser positiva.', [Axis]);
    FShape[Axis] := AShape[Axis];
    FStrides[Axis] := Count;
    Count := Count * AShape[Axis];
    if Count > MaxInt then
      raise EDelphiLMTensorError.Create('O tensor excede o limite de elementos.');
  end;
  SetLength(FData, Integer(Count));
end;

function TTensor.Rank: Integer;
begin
  Result := Length(FShape);
end;

function TTensor.Dimension(const AAxis: Integer): Integer;
begin
  if (AAxis < 0) or (AAxis >= Rank) then
    raise EDelphiLMTensorError.CreateFmt('Eixo %d inválido.', [AAxis]);
  Result := FShape[AAxis];
end;

function TTensor.ElementCount: Integer;
begin
  Result := Length(FData);
end;

function TTensor.OffsetOf(const AIndices: array of Integer): Integer;
var
  Axis: Integer;
begin
  if Length(AIndices) <> Rank then
    raise EDelphiLMTensorError.CreateFmt(
      'Esperados %d índices; recebidos %d.', [Rank, Length(AIndices)]);
  Result := 0;
  for Axis := 0 to Rank - 1 do
  begin
    if (AIndices[Axis] < 0) or (AIndices[Axis] >= FShape[Axis]) then
      raise EDelphiLMTensorError.CreateFmt(
        'Índice %d fora do eixo %d, cujo tamanho é %d.',
        [AIndices[Axis], Axis, FShape[Axis]]);
    Inc(Result, AIndices[Axis] * FStrides[Axis]);
  end;
end;

function TTensor.ValueAt(const AIndices: array of Integer): Single;
begin
  Result := FData[OffsetOf(AIndices)];
end;

procedure TTensor.SetValue(const AValue: Single;
  const AIndices: array of Integer);
begin
  FData[OffsetOf(AIndices)] := AValue;
end;

function TTensor.FlatValue(const AIndex: Integer): Single;
begin
  if (AIndex < 0) or (AIndex >= ElementCount) then
    raise EDelphiLMTensorError.CreateFmt('Índice linear %d inválido.', [AIndex]);
  Result := FData[AIndex];
end;

procedure TTensor.SetFlatValue(const AIndex: Integer; const AValue: Single);
begin
  if (AIndex < 0) or (AIndex >= ElementCount) then
    raise EDelphiLMTensorError.CreateFmt('Índice linear %d inválido.', [AIndex]);
  FData[AIndex] := AValue;
end;

procedure TTensor.Fill(const AValue: Single);
var
  Index: Integer;
begin
  for Index := 0 to High(FData) do
    FData[Index] := AValue;
end;

function DotProduct(const ALeft, ARight: TTensor): Single;
var
  Index: Integer;
begin
  if (ALeft.Rank <> 1) or (ARight.Rank <> 1) then
    raise EDelphiLMTensorError.Create('Produto escalar exige dois vetores.');
  if ALeft.ElementCount <> ARight.ElementCount then
    raise EDelphiLMTensorError.Create('Os vetores devem ter o mesmo tamanho.');
  Result := 0;
  for Index := 0 to ALeft.ElementCount - 1 do
    Result := Result + ALeft.FlatValue(Index) * ARight.FlatValue(Index);
end;

function MatrixMultiply(const ALeft, ARight: TTensor): TTensor;
var
  Column: Integer;
  Inner: Integer;
  Row: Integer;
  Sum: Single;
begin
  if (ALeft.Rank <> 2) or (ARight.Rank <> 2) then
    raise EDelphiLMTensorError.Create('Multiplicação matricial exige duas matrizes.');
  if ALeft.Dimension(1) <> ARight.Dimension(0) then
    raise EDelphiLMTensorError.CreateFmt(
      'Dimensões internas incompatíveis: %d e %d.',
      [ALeft.Dimension(1), ARight.Dimension(0)]);

  Result := TTensor.Create([ALeft.Dimension(0), ARight.Dimension(1)]);
  for Row := 0 to Result.Dimension(0) - 1 do
    for Column := 0 to Result.Dimension(1) - 1 do
    begin
      Sum := 0;
      for Inner := 0 to ALeft.Dimension(1) - 1 do
        Sum := Sum + ALeft.ValueAt([Row, Inner]) *
          ARight.ValueAt([Inner, Column]);
      Result.SetValue(Sum, [Row, Column]);
    end;
end;

end.
