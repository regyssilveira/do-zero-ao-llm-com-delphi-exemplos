{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Neural.Layers;

interface

uses
  DelphiLM.Core.Random,
  DelphiLM.Math.Tensor;

type
  TLinearLayer = class
  private
    FBias: TTensor;
    FInputSize: Integer;
    FOutputSize: Integer;
    FWeights: TTensor;
  public
    constructor Create(const AInputSize, AOutputSize: Integer;
      var ARandom: TXorShift64Star);
    destructor Destroy; override;
    function Forward(const AInput: TTensor): TTensor;
    procedure SetWeight(const AOutput, AInput: Integer;
      const AValue: Single);
    procedure SetBias(const AOutput: Integer; const AValue: Single);
    function WeightAt(const AOutput, AInput: Integer): Single;
    function BiasAt(const AOutput: Integer): Single;
    property InputSize: Integer read FInputSize;
    property OutputSize: Integer read FOutputSize;
  end;

function ReLU(const AValue: Single): Single;
function GELU(const AValue: Single): Single;

implementation

uses
  System.Math,
  System.SysUtils;

constructor TLinearLayer.Create(const AInputSize, AOutputSize: Integer;
  var ARandom: TXorShift64Star);
var
  InputIndex: Integer;
  Limit: Single;
  OutputIndex: Integer;
begin
  inherited Create;
  if AInputSize <= 0 then
    raise EDelphiLMTensorError.Create('InputSize deve ser positivo.');
  if AOutputSize <= 0 then
    raise EDelphiLMTensorError.Create('OutputSize deve ser positivo.');

  FInputSize := AInputSize;
  FOutputSize := AOutputSize;
  FWeights := TTensor.Create([FOutputSize, FInputSize]);
  FBias := TTensor.Create([FOutputSize]);

  Limit := Sqrt(6.0 / (FInputSize + FOutputSize));
  for OutputIndex := 0 to FOutputSize - 1 do
    for InputIndex := 0 to FInputSize - 1 do
      FWeights.SetValue(
        (2 * ARandom.NextSingle - 1) * Limit,
        [OutputIndex, InputIndex]);
  FBias.Fill(0);
end;

destructor TLinearLayer.Destroy;
begin
  FBias.Free;
  FWeights.Free;
  inherited;
end;

function TLinearLayer.Forward(const AInput: TTensor): TTensor;
var
  InputIndex: Integer;
  OutputIndex: Integer;
  Sum: Single;
begin
  if AInput = nil then
    raise EArgumentNilException.Create('AInput');
  if (AInput.Rank <> 1) or (AInput.ElementCount <> FInputSize) then
    raise EDelphiLMTensorError.CreateFmt(
      'A camada esperava vetor [%d].', [FInputSize]);

  Result := TTensor.Create([FOutputSize]);
  for OutputIndex := 0 to FOutputSize - 1 do
  begin
    Sum := FBias.FlatValue(OutputIndex);
    for InputIndex := 0 to FInputSize - 1 do
      Sum := Sum + FWeights.ValueAt([OutputIndex, InputIndex]) *
        AInput.FlatValue(InputIndex);
    Result.SetFlatValue(OutputIndex, Sum);
  end;
end;

procedure TLinearLayer.SetWeight(const AOutput, AInput: Integer;
  const AValue: Single);
begin
  FWeights.SetValue(AValue, [AOutput, AInput]);
end;

procedure TLinearLayer.SetBias(const AOutput: Integer;
  const AValue: Single);
begin
  FBias.SetFlatValue(AOutput, AValue);
end;

function TLinearLayer.WeightAt(const AOutput, AInput: Integer): Single;
begin
  Result := FWeights.ValueAt([AOutput, AInput]);
end;

function TLinearLayer.BiasAt(const AOutput: Integer): Single;
begin
  Result := FBias.FlatValue(AOutput);
end;

function ReLU(const AValue: Single): Single;
begin
  if AValue > 0 then
    Result := AValue
  else
    Result := 0;
end;

function GELU(const AValue: Single): Single;
const
  SqrtTwoOverPi: Single = 0.7978845608;
begin
  Result := 0.5 * AValue *
    (1 + Tanh(SqrtTwoOverPi *
      (AValue + 0.044715 * AValue * AValue * AValue)));
end;

end.
